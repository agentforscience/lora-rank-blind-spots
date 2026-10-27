"""Gradient intrinsic dimensionality at initialization.

For each LoRA target matrix W (all 7 linear types in every layer) accumulate the full-weight gradient of the SFT
loss at the pretrained weights over N examples (same prompt/target format as training), then compute its singular
value spectrum. Summaries per matrix: r50/r90/r99 (#singular values capturing that fraction of squared Frobenius
energy), entropy effective rank (Roy & Vetterli 2007), stable rank ||G||_F^2/||G||_2^2, and the energy fraction
captured by the top-8 directions (what a rank-8 first step could represent).

To fit small GPU slices, gradients are computed for one group of layers at a time (repeated forward/backward passes).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import load_task  # noqa: E402
from train_eval import reserve_gpu_memory, MODELS, TARGETS, encode, collate, lm_loss, set_seed  # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spectrum_stats(G):
    s = torch.linalg.svdvals(G.float().cuda()).double()
    e = s ** 2
    tot = e.sum()
    c = torch.cumsum(e, 0) / tot
    p = s / s.sum()
    ent = -(p[p > 0] * torch.log(p[p > 0])).sum()
    out = {f"r{int(q*100)}": int((c < q).sum().item()) + 1 for q in (0.5, 0.9, 0.99)}
    out["erank"] = float(torch.exp(ent))
    out["stable_rank"] = float(tot / e[0])
    out["top8_energy"] = float(c[min(7, len(c) - 1)])
    out["dim"] = min(G.shape)
    return out, s[:256].float().cpu().numpy().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.5B")
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--micro_bs", type=int, default=4)
    ap.add_argument("--layer_groups", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_path = os.path.join(WS, "results", "grad_id", f"{a.model}_{a.task}_n{a.n}_s{a.seed}.json")
    if os.path.exists(out_path):
        print("EXISTS", out_path)
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import fcntl
    lock = open(out_path + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("LOCKED", out_path)
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reserve_gpu_memory()
    set_seed(a.seed)
    T = load_task(a.task)
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(False)
    ex = random.Random(a.seed).sample(T["train"], a.n)
    data = [encode(tok, e) for e in ex]
    layers = model.model.layers
    nL = len(layers)
    groups = np.array_split(np.arange(nL), a.layer_groups)
    per_matrix = []
    spectra = {}
    for g in groups:
        mats = {}
        for li in g:
            for name in TARGETS:
                mod = getattr(layers[li].self_attn, name, None) or getattr(layers[li].mlp, name)
                mod.weight.requires_grad_(True)
                mats[(int(li), name)] = mod.weight
        acc = {k: torch.zeros_like(w, dtype=torch.float32) for k, w in mats.items()}
        model.train()  # dropout is 0 in Qwen2.5; train mode only matters for checkpointing
        ntok = 0
        for i in range(0, len(data), a.micro_bs):
            ids, lab, att = [x.cuda() for x in collate(data[i:i + a.micro_bs], tok.pad_token_id)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, c = lm_loss(model, ids, lab, att, reduction="sum")
            loss.backward()
            ntok += c.item()
            for k, w in mats.items():
                acc[k] += w.grad.float()
                w.grad = None
        for k, w in mats.items():
            w.requires_grad_(False)
            G = acc[k] / ntok  # gradient of the mean token loss
            st, sv = spectrum_stats(G)
            st.update({"layer": k[0], "module": k[1], "fro": float(G.norm())})
            per_matrix.append(st)
            spectra[f"{k[0]}.{k[1]}"] = sv
        del acc
        torch.cuda.empty_cache()
        print(f"group {g.tolist()} done", flush=True)
    keys = ["r50", "r90", "r99", "erank", "stable_rank", "top8_energy"]
    summary = {k: float(np.median([m[k] for m in per_matrix])) for k in keys}
    # Frobenius-weighted mean (matrices with larger gradients dominate the first update)
    w = np.array([m["fro"] for m in per_matrix]) ** 2
    summary.update({f"{k}_fw": float(np.sum(w * np.array([m[k] for m in per_matrix])) / w.sum()) for k in keys})
    res = {"model": a.model, "task": a.task, "n": a.n, "seed": a.seed, "summary": summary,
           "per_matrix": per_matrix, "spectra": spectra}
    with open(out_path, "w") as f:
        json.dump(res, f)
    print("DONE", a.model, a.task, json.dumps(summary))


if __name__ == "__main__":
    main()
