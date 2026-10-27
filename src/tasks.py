"""Task definitions: load raw data, format as (prompt, target) pairs for causal-LM SFT, and evaluation sets.

Every task returns:
  train:   list of dicts {prompt, target}                (N_TRAIN examples, fixed data seed)
  heldout: list of dicts {prompt, target}                (N_HELDOUT examples from the train distribution, for NLL)
  eval:    list of dicts with task-specific fields       (benchmark evaluation set)
and a `kind`: "cls" (label-restricted scoring), "gsm8k" (generation + exact match), "code" (generation + execution).
"""
import os
import re
import random
import pandas as pd

HUB = "/data/transformers/hub"
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRAIN = 4000
N_HELDOUT = 300
N_EVAL_CLS = 1000
N_EVAL_GSM = 500
DATA_SEED = 0

P = {
    "sst2": f"{HUB}/datasets--stanfordnlp--sst2/snapshots/8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb/data/{{}}-00000-of-00001.parquet",
    "emotion": f"{HUB}/datasets--dair-ai--emotion/snapshots/cab853a1dbdf4c42c2b3ef2173804746df8825fe/split/{{}}-00000-of-00001.parquet",
    "ag_news": f"{HUB}/datasets--fancyzhx--ag_news/snapshots/eb185aade064a813bc0b7f42de02595523103ca4/data/{{}}-00000-of-00001.parquet",
    "boolq": f"{HUB}/datasets--google--boolq/snapshots/35b264d03638db9f4ce671b711558bf7ff0f80d5/data/{{}}-00000-of-00001.parquet",
    "gsm8k": f"{HUB}/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/{{}}-00000-of-00001.parquet",
    "mbpp": f"{HUB}/datasets--google-research-datasets--mbpp/snapshots/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/{{}}-00000-of-00001.parquet",
    "code_train": f"{WS}/datasets/raw/code_alpaca_py.parquet",
}

LABELS = {
    "sst2": ["negative", "positive"],
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "ag_news": ["World", "Sports", "Business", "Technology"],
    "boolq": ["no", "yes"],
}
TASKS = ["sst2", "emotion", "ag_news", "boolq", "gsm8k", "code"]


def _split(df, n_train, n_heldout):
    idx = list(range(len(df)))
    random.Random(DATA_SEED).shuffle(idx)
    return df.iloc[idx[:n_train]], df.iloc[idx[n_train:n_train + n_heldout]]


def _sample(df, n):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=DATA_SEED)


def cls_prompt(task, row):
    if task == "sst2":
        return f"Review: {row['sentence'].strip()}\nSentiment:"
    if task == "emotion":
        return f"Text: {row['text'].strip()}\nEmotion:"
    if task == "ag_news":
        return f"Article: {row['text'].strip()}\nTopic:"
    if task == "boolq":
        passage = row["passage"].strip()
        if len(passage) > 1500:
            passage = passage[:1500] + "..."
        return f"Passage: {passage}\nQuestion: {row['question'].strip()}?\nAnswer:"
    raise ValueError(task)


def cls_label(task, row):
    if task == "boolq":
        return int(bool(row["answer"]))
    return int(row["label"])


def strip_calc(ans):
    """Remove GSM8K calculator annotations <<a*b=c>>."""
    return re.sub(r"<<.*?>>", "", ans)


def gsm_prompt(q):
    return f"Question: {q.strip()}\nAnswer:"


def code_prompt(instruction, inp=""):
    body = instruction.strip()
    if inp and str(inp).strip():
        body += "\n" + str(inp).strip()
    return f"### Instruction:\n{body}\n\n### Response:\n"


def load_task(task):
    if task in LABELS:
        train_split = "train"
        eval_split = {"sst2": "validation", "emotion": "test", "ag_news": "test", "boolq": "validation"}[task]
        tr = pd.read_parquet(P[task].format(train_split))
        ev = pd.read_parquet(P[task].format(eval_split))
        tr_s, ho_s = _split(tr, N_TRAIN, N_HELDOUT)
        fmt = lambda df: [{"prompt": cls_prompt(task, r), "target": " " + LABELS[task][cls_label(task, r)]}
                          for _, r in df.iterrows()]
        ev_s = _sample(ev, N_EVAL_CLS)
        evl = [{"prompt": cls_prompt(task, r), "label": cls_label(task, r)} for _, r in ev_s.iterrows()]
        return {"kind": "cls", "labels": LABELS[task], "train": fmt(tr_s), "heldout": fmt(ho_s), "eval": evl}
    if task == "gsm8k":
        tr = pd.read_parquet(P["gsm8k"].format("train"))
        ev = pd.read_parquet(P["gsm8k"].format("test"))
        tr_s, ho_s = _split(tr, N_TRAIN, N_HELDOUT)
        fmt = lambda df: [{"prompt": gsm_prompt(r["question"]), "target": " " + strip_calc(r["answer"]).strip()}
                          for _, r in df.iterrows()]
        ev = _sample(ev, N_EVAL_GSM)
        evl = [{"prompt": gsm_prompt(r["question"]), "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
               for _, r in ev.iterrows()]
        return {"kind": "gsm8k", "train": fmt(tr_s), "heldout": fmt(ho_s), "eval": evl}
    if task == "code":
        tr = pd.read_parquet(P["code_train"])
        tr_s, ho_s = _split(tr, N_TRAIN, N_HELDOUT)
        fmt = lambda df: [{"prompt": code_prompt(r["instruction"], r["input"]), "target": r["output"].strip()}
                          for _, r in df.iterrows()]
        ev = pd.read_parquet(P["mbpp"].format("test"))
        evl = [{"prompt": code_prompt(f"{r['text'].strip()}\nYour code should pass this test:\n{r['test_list'][0]}"),
                "tests": list(r["test_list"]), "setup": r["test_setup_code"], "task_id": int(r["task_id"])}
               for _, r in ev.iterrows()]
        return {"kind": "code", "train": fmt(tr_s), "heldout": fmt(ho_s), "eval": evl}
    raise ValueError(task)


def extract_gsm_answer(text):
    """Final numeric answer: after '####' if present, else last number in the text."""
    text = text.split("\nQuestion:")[0]
    if "####" in text:
        m = re.findall(r"-?\d[\d,]*\.?\d*", text.split("####")[-1])
        if m:
            return m[0].replace(",", "").rstrip(".")
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def gsm_correct(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def extract_code(text):
    text = text.split("\n###")[0]
    m = re.search(r"```(?:python)?\n(.*?)(```|$)", text, re.S)
    return m.group(1) if m else text
