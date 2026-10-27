"""Render markdown tables from results/*.csv into REPORT.md placeholders (writes results/tables.md too)."""
import json
import os

import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(WS, "results")
RANKS = [0, 1, 2, 4, 8, 16, 32, 64, 128]


def md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def rank_table():
    s = pd.read_csv(os.path.join(R, "summary_table.csv"))
    rows = []
    for (m, t), g in s.groupby(["model", "task"], sort=False):
        g = g.set_index("rank")
        best = g.loc[g.index > 0, "metric"].idxmax()
        row = {"model": m, "task": t}
        for r in RANKS:
            if r in g.index:
                v = f"{100 * g.loc[r, 'metric']:.1f}"
                row["r0 (zero-shot)" if r == 0 else f"r{r}"] = f"**{v}**" if r == best else v
            else:
                row["r0 (zero-shot)" if r == 0 else f"r{r}"] = "–"
        rows.append(row)
    df = pd.DataFrame(rows)
    order = {"sst2": 0, "emotion": 1, "ag_news": 2, "boolq": 3, "gsm8k": 4, "code": 5}
    df["o"] = df.task.map(order)
    return md(df.sort_values(["model", "o"]).drop(columns="o"))


def knee_table():
    k = pd.read_csv(os.path.join(R, "knees.csv"))
    c = pd.read_csv(os.path.join(R, "knees_coarse.csv"))
    rows = []
    for _, r in c.iterrows():
        f = k[(k.model == r.model) & (k.task == r.task)]
        rows.append({"model": r.model, "task": r.task,
                     "best rank (full)": int(f.best_rank.iloc[0]) if len(f) else "–",
                     "knee (full)": int(f.knee.iloc[0]) if len(f) else "–",
                     "knee 1pp (full)": int(f.knee_1pp.iloc[0]) if len(f) else "–",
                     "NLL knee95 (full)": int(f.knee_nll95.iloc[0]) if len(f) else "–",
                     "best rank (1/8/32/128)": int(r.best_rank), "knee (1/8/32/128)": int(r.knee),
                     "best %": f"{100 * (f.best_metric.iloc[0] if len(f) else r.best_metric):.1f}",
                     "r8 %": f"{100 * r.r8_metric:.1f}", "seeds": int(r.n_seeds)})
    return md(pd.DataFrame(rows))


def gap_table():
    g = pd.read_csv(os.path.join(R, "gaps.csv"))
    g = g.assign(gap=lambda d: (100 * d.gap_naive).round(1),
                 CI=lambda d: [f"[{100*a:.1f}, {100*b:.1f}]" for a, b in zip(d.ci_lo, d.ci_hi)],
                 crossfit=lambda d: [("–" if pd.isna(x) else f"{100*x:.1f}") for x in d.get("gap_crossfit", np.nan)],
                 best_above8=lambda d: (100 * d.gap_best_above8).round(1),
                 p_holm=lambda d: d.p_holm.round(3))
    return md(g[["model", "task", "best_rank", "gap", "CI", "crossfit", "best_above8", "p_holm"]]
              .rename(columns={"gap": "best − r8 (pp)", "crossfit": "cross-fitted (pp)",
                               "best_above8": "best r>8 − r8 (pp)"}))


def lr_table():
    t = pd.read_csv(os.path.join(R, "lr_sweep_metric.csv"))
    t = t.rename(columns={c: f"r{c}" for c in t.columns if c.isdigit()})
    for c in t.columns[2:]:
        t[c] = t[c].map(lambda v: "–" if pd.isna(v) else f"{100*v:.1f}")
    b = pd.read_csv(os.path.join(R, "lr_sweep_best_over_lr.csv"))
    b = b.rename(columns={c: f"r{c}" for c in b.columns if c.isdigit()})
    for c in b.columns[1:]:
        b[c] = b[c].map(lambda v: f"{100*v:.1f}")
    b.insert(1, "lr", "best over lr")
    t["lr"] = t["lr"].map(lambda v: f"{v:g}")
    return md(pd.concat([t, b]))


def pred_table():
    p = json.load(open(os.path.join(R, "predictor.json")))
    rows = []
    for key in ["knee~r90", "knee~r99", "knee~erank", "knee~top8_energy", "knee~stable_rank", "knee~zero_shot",
                "best_rank~r90", "best_rank~erank", "knee_nll95~r90", "knee_nll95~r99"]:
        if key in p:
            v = p[key]
            rows.append({"target ~ predictor": key, "Spearman ρ": f"{v['rho']:.2f}", "perm. p": f"{v['p_perm']:.3f}",
                         "n": v["n"], "within 2× (LOO-calibrated)": "–" if v["within2_loo"] is None else f"{v['within2_loo']:.2f}"})
    for key in ["knee~const8", "best_rank~const8"]:
        rows.append({"target ~ predictor": key + " (baseline: always r=8)", "Spearman ρ": "–", "perm. p": "–",
                     "n": "–", "within 2× (LOO-calibrated)": f"{p[key]['within2']:.2f}"})
    return md(pd.DataFrame(rows)) + ("\n\n(top8_energy and zero-shot accuracy are sign-flipped so that a positive ρ "
                                     "means 'predicts higher rank'.)")


if __name__ == "__main__":
    tabs = {"RANK_TABLE_PLACEHOLDER": rank_table(), "KNEE_TABLE_PLACEHOLDER": knee_table(),
            "GAP_TABLE_PLACEHOLDER": gap_table(), "LR_TABLE_PLACEHOLDER": lr_table(),
            "PRED_PLACEHOLDER": pred_table()}
    with open(os.path.join(R, "tables.md"), "w") as f:
        for k, v in tabs.items():
            f.write(f"## {k}\n\n{v}\n\n")
    print(open(os.path.join(R, "tables.md")).read())
