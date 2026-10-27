"""Analysis of the rank sweep: knee/best ranks, rank-8 gaps, model-size comparison, gradient-ID predictor.

Outputs: results/summary_table.csv, results/knees.csv, results/gaps.csv, results/predictor.json, figures/*.png
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(WS, "results")
FIG = os.path.join(WS, "figures")
os.makedirs(FIG, exist_ok=True)
RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
TASKS = ["sst2", "emotion", "ag_news", "boolq", "gsm8k", "code"]
MODELS = ["0.5B", "1.5B"]
MAIN_LR = 2e-4
rng = np.random.default_rng(0)


def load_runs():
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "runs", "*.json"))):
        d = json.load(open(f))
        rows.append(d)
    return pd.DataFrame(rows)


def item_matrix(df, model, task, rank, lr=MAIN_LR, rslora=False):
    """(n_seeds, n_items) correctness matrix for one cell."""
    sub = df[(df.model == model) & (df.task == task) & (df["rank"] == rank) & np.isclose(df.lr, lr) & (df.rslora == rslora)]
    sub = sub.sort_values("seed")
    if len(sub) == 0:
        return None, []
    return np.array(sub.correct.tolist(), dtype=float), sub.seed.tolist()


def knee_from_curve(means, ses, ranks, min_delta=0.005):
    """Smallest rank whose mean is within delta of the best; delta = max(min_delta, SE of best)."""
    best_i = int(np.nanargmax(means))
    delta = max(min_delta, ses[best_i])
    for r, m in zip(ranks, means):
        if m >= means[best_i] - delta:
            return r, ranks[best_i], delta
    return ranks[best_i], ranks[best_i], delta


def main():
    df = load_runs()
    df = df[~df.rslora]
    lr_sweep(df)
    spectrum_plot()
    main_df = df[np.isclose(df.lr, MAIN_LR)]
    # ---------- summary table ----------
    summ = (main_df.groupby(["model", "task", "rank"])
            .agg(metric=("metric", "mean"), metric_sd=("metric", "std"), nll=("heldout_nll", "mean"),
                 n_seeds=("seed", "nunique"), n_trainable=("n_trainable", "first"), train_time=("train_time_s", "mean"))
            .reset_index())
    summ.to_csv(os.path.join(R, "summary_table.csv"), index=False)

    # ---------- knees ----------
    def knee_rows(ladder):
        rows = []
        for model in MODELS:
            for task in TASKS:
                s = summ[(summ.model == model) & (summ.task == task) & (summ["rank"] > 0)].set_index("rank")
                if not set(ladder) <= set(s.index):
                    continue
                means, ses = [], []
                for r in ladder:
                    M, _ = item_matrix(main_df, model, task, r)
                    pm = M.mean(0)  # per-item mean over seeds
                    means.append(pm.mean())
                    ses.append(pm.std(ddof=1) / np.sqrt(len(pm)))  # SE of the item-mean
                means = np.array(means)
                k05, best, delta = knee_from_curve(means, ses, ladder, 0.005)
                k1, _, _ = knee_from_curve(means, [0] * len(ladder), ladder, 0.01)
                nll = s.loc[ladder, "nll"].values
                z = summ[(summ.model == model) & (summ.task == task) & (summ["rank"] == 0)]
                nll0 = z.nll.values[0] if len(z) else nll.max()
                gain = nll0 - nll
                rows.append(dict(model=model, task=task, best_rank=best, knee=k05, knee_delta=delta, knee_1pp=k1,
                                 knee_nll95=ladder[int(np.argmax(gain >= 0.95 * gain.max()))],
                                 knee_nll99=ladder[int(np.argmax(gain >= 0.99 * gain.max()))],
                                 best_nll_rank=ladder[int(np.argmin(nll))],
                                 best_metric=means.max(), r8_metric=means[ladder.index(8)],
                                 zero_shot=z.metric.values[0] if len(z) else np.nan,
                                 n_seeds=int(s.loc[ladder, "n_seeds"].min())))
        return rows
    coarse = pd.DataFrame(knee_rows([1, 8, 32, 128]))
    coarse.to_csv(os.path.join(R, "knees_coarse.csv"), index=False)
    print("coarse-ladder knees (1/8/32/128):\n", coarse.to_string())
    knees = knee_rows(RANKS)
    knees = pd.DataFrame(knees)
    knees.to_csv(os.path.join(R, "knees.csv"), index=False)
    print(knees.to_string())

    # ---------- rank-8 gaps (paired bootstrap over items) ----------
    gaps = []
    for _, k in knees.iterrows():
        M8, seeds8 = item_matrix(main_df, k.model, k.task, 8)
        Mb, seedsb = item_matrix(main_df, k.model, k.task, k.best_rank)
        d = Mb.mean(0) - M8.mean(0)
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        row = dict(model=k.model, task=k.task, best_rank=k.best_rank, gap_naive=d.mean(),
                   ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                   p_boot=2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        # cross-fitted gap: pick best rank on one seed, measure on the other (removes winner's-curse bias)
        if all(len(item_matrix(main_df, k.model, k.task, r)[1]) >= 2 for r in RANKS):
            cf = []
            for sel, ev in [(0, 1), (1, 0)]:
                m_sel = [item_matrix(main_df, k.model, k.task, r)[0][sel].mean() for r in RANKS]
                rb = RANKS[int(np.argmax(m_sel))]
                cf.append(item_matrix(main_df, k.model, k.task, rb)[0][ev].mean() - M8[ev].mean())
            row["gap_crossfit"] = float(np.mean(cf))
        # gap of the best *larger* rank (is there anything to gain above 8?)
        hi = [r for r in RANKS if r > 8]
        mh = [item_matrix(main_df, k.model, k.task, r)[0].mean() for r in hi]
        row["gap_best_above8"] = max(mh) - M8.mean()
        gaps.append(row)
    gaps = pd.DataFrame(gaps)
    p = gaps.p_boot.values
    order = np.argsort(p)
    holm = np.empty_like(p)
    running = 0
    for i, j in enumerate(order):
        running = max(running, min(1, (len(p) - i) * p[j]))
        holm[j] = running
    gaps["p_holm"] = holm
    gaps.to_csv(os.path.join(R, "gaps.csv"), index=False)
    print(gaps.to_string())

    # ---------- figures: metric and NLL vs rank ----------
    fig, axes = plt.subplots(2, 6, figsize=(22, 7))
    colors = {"0.5B": "#1f77b4", "1.5B": "#d62728"}
    for j, task in enumerate(TASKS):
        for model in MODELS:
            s = summ[(summ.model == model) & (summ.task == task) & (summ["rank"] > 0)].sort_values("rank")
            if len(s) == 0:
                continue
            per_seed = main_df[(main_df.model == model) & (main_df.task == task) & (main_df["rank"] > 0)]
            axes[0, j].plot(s["rank"], 100 * s.metric, "o-", color=colors[model], label=model)
            axes[0, j].scatter(per_seed["rank"], 100 * per_seed.metric, color=colors[model], s=8, alpha=0.4)
            z = summ[(summ.model == model) & (summ.task == task) & (summ["rank"] == 0)]
            if len(z):
                axes[0, j].axhline(100 * z.metric.values[0], color=colors[model], ls=":", lw=1)
            axes[1, j].plot(s["rank"], s.nll, "o-", color=colors[model], label=model)
            kk = knees[(knees.model == model) & (knees.task == task)]
            if len(kk):
                axes[0, j].axvline(kk.knee.values[0], color=colors[model], ls="--", lw=1, alpha=0.6)
        for i in range(2):
            axes[i, j].set_xscale("log", base=2)
            axes[i, j].set_xticks(RANKS)
            axes[i, j].set_xticklabels(RANKS)
            axes[i, j].axvline(8, color="gray", lw=0.8, alpha=0.4)
        axes[0, j].set_title(task)
        axes[1, j].set_xlabel("LoRA rank r")
    axes[0, 0].set_ylabel("eval metric (%)\n(dotted = zero-shot, dashed = knee)")
    axes[1, 0].set_ylabel("held-out target NLL (nats/token)")
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "rank_curves.png"), dpi=130)
    plt.close()

    # ---------- gradient ID predictor ----------
    gid = []
    for f in glob.glob(os.path.join(R, "grad_id", "*.json")):
        d = json.load(open(f))
        gid.append(dict(model=d["model"], task=d["task"], **d["summary"]))
    if gid:
        gid = pd.DataFrame(gid)
        m = knees.merge(gid, on=["model", "task"])
        pred = {}
        preds = ["r50", "r90", "r99", "erank", "stable_rank", "top8_energy", "r90_fw", "erank_fw", "zero_shot"]
        for target in ["knee", "best_rank", "knee_nll95", "knee_nll99"]:
            for pc in preds:
                x = m[pc].values.astype(float)
                if pc in ("top8_energy", "zero_shot"):
                    x = -x  # less top-8 energy / lower zero-shot accuracy -> expected higher rank
                y = np.log2(m[target].values.astype(float))
                if len(m) < 4 or np.std(y) == 0:
                    continue
                rho, _ = stats.spearmanr(x, y)
                perm = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(5000)])
                p = (np.abs(perm) >= abs(rho)).mean()
                # factor-of-2 accuracy with a leave-one-out calibrated log-linear map log2(knee) = a + b*log2(x)
                within2 = None
                if pc not in ("top8_energy", "zero_shot"):
                    lx = np.log2(np.maximum(m[pc].values.astype(float), 1e-9))
                    hits = []
                    for i in range(len(m)):
                        msk = np.arange(len(m)) != i
                        a = np.mean(y[msk] - lx[msk])  # slope fixed at 1: pure scale calibration
                        hits.append(abs((a + lx[i]) - y[i]) <= 1)
                    within2 = float(np.mean(hits))
                pred[f"{target}~{pc}"] = dict(rho=float(rho), p_perm=float(p), n=len(m), within2_loo=within2)
        # baseline: always predict rank 8
        for target in ["knee", "best_rank", "knee_nll95"]:
            pred[f"{target}~const8"] = dict(within2=float(np.mean(np.abs(np.log2(m[target]) - 3) <= 1)))
        json.dump(pred, open(os.path.join(R, "predictor.json"), "w"), indent=1)
        m.to_csv(os.path.join(R, "knees_with_gradid.csv"), index=False)
        for k, v in pred.items():
            print(k, v)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, (pc, target) in zip(axes, [("r90", "knee"), ("r90", "knee_nll95"), ("erank", "knee")]):
            for model, mk in [("0.5B", "o"), ("1.5B", "s")]:
                mm = m[m.model == model]
                ax.scatter(mm[pc], mm[target], marker=mk, label=model, s=50)
                for _, r in mm.iterrows():
                    ax.annotate(r.task, (r[pc], r[target]), fontsize=8, xytext=(3, 3), textcoords="offset points")
            ax.set_yscale("log", base=2)
            ax.set_xlabel(f"gradient {pc} at init (median over matrices)")
            ax.set_ylabel(target)
            key = f"{target}~{pc}"
            if key in pred:
                ax.set_title(f"Spearman rho={pred[key]['rho']:.2f} (p={pred[key]['p_perm']:.2f})")
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "gradid_vs_knee.png"), dpi=130)
        plt.close()


def spectrum_plot():
    """Median (over target matrices) cumulative gradient energy captured by the top-k directions, per task."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        for task in TASKS:
            f = os.path.join(R, "grad_id", f"{model}_{task}_n128_s0.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            curves = []
            for sv in d["spectra"].values():
                e = np.array(sv, dtype=float) ** 2
                # energy relative to the full spectrum is not stored beyond 256 values; per-matrix top8_energy etc.
                # were computed on the full spectrum. Here: cumulative share within the top-256 values.
                c = np.cumsum(e) / e.sum()
                curves.append(np.pad(c, (0, 256 - len(c)), constant_values=1.0))
            med = np.median(np.array(curves), axis=0)
            ax.plot(np.arange(1, 257), 1 - med, label=task)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("k (number of top singular directions)")
        ax.set_title(f"Qwen2.5-{model}: gradient energy NOT captured by top-k")
        ax.axvline(8, color="gray", lw=0.8)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("1 - cumulative energy (median over matrices)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "gradient_spectra.png"), dpi=130)
    plt.close()


def lr_sweep(df):
    """Exp 4: seed-0 metric for ranks {2,8,32,128} x lr; best-over-lr curve per rank."""
    sub = df[(df.model == "0.5B") & (df.seed == 0) & (df.task.isin(["code", "gsm8k", "emotion"]))
             & (df["rank"].isin([2, 8, 32, 128]))]
    tab = sub.pivot_table(index=["task", "lr"], columns="rank", values="metric")
    nll = sub.pivot_table(index=["task", "lr"], columns="rank", values="heldout_nll")
    tab.to_csv(os.path.join(R, "lr_sweep_metric.csv"))
    nll.to_csv(os.path.join(R, "lr_sweep_nll.csv"))
    print((100 * tab).round(1).to_string())
    print(nll.round(4).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, task in zip(axes, ["code", "gsm8k", "emotion"]):
        if task not in tab.index.get_level_values(0):
            continue
        t = tab.loc[task]
        for lr, row in t.iterrows():
            ax.plot(row.index, 100 * row.values, "o-", label=f"lr={lr:g}", alpha=0.8)
        ax.plot(t.columns, 100 * t.max(axis=0).values, "k--", lw=2, label="best lr per rank")
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 8, 32, 128])
        ax.set_xticklabels([2, 8, 32, 128])
        ax.set_title(f"{task} (Qwen2.5-0.5B, seed 0)")
        ax.set_xlabel("LoRA rank r")
        ax.set_ylabel("eval metric (%)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "lr_sweep.png"), dpi=130)
    plt.close()
    best = tab.groupby(level=0).max()
    best.to_csv(os.path.join(R, "lr_sweep_best_over_lr.csv"))
    print("best over lr:\n", (100 * best).round(1).to_string())


if __name__ == "__main__":
    main()
