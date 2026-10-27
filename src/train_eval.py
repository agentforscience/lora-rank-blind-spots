"""Train one LoRA adapter (model, task, rank, seed, lr) and evaluate it.

Writes results/runs/<tag>.json with config, training curve, held-out NLL and per-item eval correctness.
rank 0 = zero-shot base model (no training).
"""
import argparse
import fcntl
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import load_task, extract_gsm_answer, gsm_correct, extract_code  # noqa: E402
from code_exec import run_tests  # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"0.5B": "Qwen/Qwen2.5-0.5B", "1.5B": "Qwen/Qwen2.5-1.5B", "3B": "Qwen/Qwen2.5-3B"}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN = 512
CE_CHUNK = 512


def reserve_gpu_memory():
    """Grab RESERVE_GB of GPU memory into PyTorch's caching allocator right away, so co-tenants on the shared GPU
    cannot take it while the job is running (the block is freed to the cache, not to the driver, and reused)."""
    gb = float(os.environ.get("RESERVE_GB", "0"))
    if gb > 0:
        x = torch.empty(int(gb * 1e9), dtype=torch.uint8, device="cuda")
        del x


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def encode(tok, ex):
    """Tokenize prompt+target+eos; labels only on target/eos tokens. Truncate from the left of the prompt if needed."""
    p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    t = tok(ex["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    t = t[:MAX_LEN - 16]
    p = p[-(MAX_LEN - len(t)):]
    return p + t, [-100] * len(p) + t


def collate(batch, pad_id):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    return ids, lab, att


def _ce(head, h, y):
    return F.cross_entropy(head(h).float(), y, reduction="sum")


def lm_loss(model, ids, lab, att, reduction="mean"):
    """Causal-LM loss computed only at labelled positions (avoids materialising full-vocab logits for prompts)."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    h = h[:, :-1]
    y = lab[:, 1:]
    m = y != -100
    hs, ys = h[m], y[m]
    loss = 0.0
    for i in range(0, hs.shape[0], CE_CHUNK):  # chunked + checkpointed: never holds more than CE_CHUNK x vocab logits
        hc, yc = hs[i:i + CE_CHUNK], ys[i:i + CE_CHUNK]
        if torch.is_grad_enabled():
            loss = loss + checkpoint(_ce, inner.lm_head, hc, yc, use_reentrant=False)
        else:
            loss = loss + _ce(inner.lm_head, hc, yc)
    if reduction == "mean":
        return loss / m.sum()
    return loss, m.sum()


@torch.no_grad()
def heldout_nll(model, data, pad_id, bs=8):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), bs):
        ids, lab, att = [x.cuda() for x in collate(data[i:i + bs], pad_id)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, c = lm_loss(model, ids, lab, att, reduction="sum")
        tot += l.item()
        n += c.item()
    return tot / n


@torch.no_grad()
def eval_cls(model, tok, items, labels, bs=16):
    model.eval()
    lab_ids = [tok(" " + l, add_special_tokens=False)["input_ids"][0] for l in labels]
    assert len(set(lab_ids)) == len(lab_ids), "label first tokens collide"
    tok.padding_side = "left"
    correct = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        enc = tok([c["prompt"] for c in chunk], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to("cuda")
        inner = model.get_base_model() if hasattr(model, "get_base_model") else model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = inner.model(**enc).last_hidden_state[:, -1]
            logits = inner.lm_head(h).float()[:, lab_ids]
        pred = logits.argmax(-1).tolist()
        correct += [int(p == c["label"]) for p, c in zip(pred, chunk)]
    return correct, None


@torch.no_grad()
def generate(model, tok, prompts, max_new, stop, bs):
    model.eval()
    tok.padding_side = "left"
    outs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))  # length-sorted batches
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id,
                               eos_token_id=tok.eos_token_id, stop_strings=stop, tokenizer=tok)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outs[i] = t
    return outs


def evaluate(model, tok, T, gen_bs):
    if T["kind"] == "cls":
        return eval_cls(model, tok, T["eval"], T["labels"])
    if T["kind"] == "gsm8k":
        outs = generate(model, tok, [e["prompt"] for e in T["eval"]], 320, ["\nQuestion:"], gen_bs)
        corr = [int(gsm_correct(extract_gsm_answer(o), e["answer"])) for o, e in zip(outs, T["eval"])]
        return corr, outs
    if T["kind"] == "code":
        outs = generate(model, tok, [e["prompt"] for e in T["eval"]], 384, ["\n###"], gen_bs)
        codes = [extract_code(o) for o in outs]
        corr = [int(x) for x in run_tests(codes, T["eval"])]
        return corr, outs
    raise ValueError(T["kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.5B")
    ap.add_argument("--task", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha_mult", type=float, default=2.0, help="alpha = alpha_mult * r (standard scaling)")
    ap.add_argument("--rslora", action="store_true", help="use alpha/sqrt(r) scaling instead of alpha/r")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16, help="effective batch size")
    ap.add_argument("--micro_bs", type=int, default=0)
    ap.add_argument("--grad_ckpt", type=int, default=-1)
    ap.add_argument("--out_dir", default=os.path.join(WS, "results", "runs"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    tag = a.tag or f"{a.model}_{a.task}_r{a.rank}_s{a.seed}_lr{a.lr:g}" + ("_rs" if a.rslora else "")
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, tag + ".json")
    if os.path.exists(out_path):
        print("EXISTS", out_path)
        return
    lock = open(out_path + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("LOCKED", out_path)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    reserve_gpu_memory()
    t0 = time.time()
    set_seed(a.seed)
    T = load_task(a.task)
    path = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda()
    long_task = a.task in ("gsm8k", "code", "boolq")
    grad_ckpt = a.grad_ckpt if a.grad_ckpt >= 0 else 1
    micro = a.micro_bs or (4 if long_task else 16)
    gen_bs = (64 if a.model == "0.5B" else 32) // (4 if a.task == "code" else 1)

    curve = []
    if a.rank > 0:
        cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha_mult * a.rank, lora_dropout=0.05, target_modules=TARGETS,
                         use_rslora=a.rslora, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        adapter_dir = os.path.join(WS, "results", "adapters", tag)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if os.path.exists(os.path.join(adapter_dir, "state.pt")):
            # resume after an eval-time failure: reuse the trained adapter
            ck = torch.load(os.path.join(adapter_dir, "state.pt"))
            missing = model.load_state_dict(ck["lora"], strict=False)
            assert not [k for k in missing.unexpected_keys], missing.unexpected_keys
            curve, steps, t_resume = ck["curve"], ck["steps"], ck["train_time"]
            print("loaded trained adapter from", adapter_dir, flush=True)
        else:
            t_resume = None
    if a.rank > 0 and t_resume is None:
        if grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
        data = [encode(tok, ex) for ex in T["train"]]
        accum = a.bs // micro
        steps = int(math.ceil(len(data) * a.epochs / a.bs))
        warm = max(1, int(0.03 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
        rng = random.Random(a.seed)
        order = []
        while len(order) < steps * a.bs:
            perm = list(range(len(data)))
            rng.shuffle(perm)
            order += perm
        model.train()
        run_loss = 0.0
        for step in range(steps):
            for k in range(accum):
                bidx = order[step * a.bs + k * micro: step * a.bs + (k + 1) * micro]
                ids, lab, att = [x.cuda() for x in collate([data[i] for i in bidx], tok.pad_token_id)]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = lm_loss(model, ids, lab, att) / accum
                loss.backward()
                run_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if (step + 1) % 10 == 0 or step == steps - 1:
                curve.append((step + 1, run_loss / 10))
                if (step + 1) % 50 == 0:
                    print(f"step {step+1}/{steps} loss {run_loss/10:.4f} {time.time()-t0:.0f}s", flush=True)
                run_loss = 0.0
        if grad_ckpt:
            model.gradient_checkpointing_disable()
        os.makedirs(adapter_dir, exist_ok=True)
        lora_sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
        torch.save({"lora": lora_sd, "curve": curve, "steps": steps, "train_time": time.time() - t0},
                   os.path.join(adapter_dir, "state.pt"))
    if a.rank == 0:
        n_trainable, steps = 0, 0
    t_train = t_resume if (a.rank > 0 and t_resume is not None) else time.time() - t0
    nll = heldout_nll(model, [encode(tok, ex) for ex in T["heldout"]], tok.pad_token_id)
    corr, outs = evaluate(model, tok, T, gen_bs)
    res = {
        "tag": tag, "model": a.model, "task": a.task, "rank": a.rank, "seed": a.seed, "lr": a.lr,
        "alpha_mult": a.alpha_mult, "rslora": a.rslora, "epochs": a.epochs, "bs": a.bs, "steps": steps,
        "n_trainable": n_trainable, "heldout_nll": nll, "metric": float(np.mean(corr)), "n_eval": len(corr),
        "correct": corr, "curve": curve, "train_time_s": t_train, "total_time_s": time.time() - t0,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    with open(out_path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out_path + ".tmp", out_path)
    if outs is not None:
        os.makedirs(os.path.join(WS, "results", "outputs"), exist_ok=True)
        with open(os.path.join(WS, "results", "outputs", tag + ".json"), "w") as f:
            json.dump(outs, f)
    print(f"DONE {tag} metric={res['metric']:.4f} nll={nll:.4f} train={t_train:.0f}s total={res['total_time_s']:.0f}s "
          f"peak={res['peak_mem_gb']:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
