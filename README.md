# LoRA Rank Selection Blind Spots

A controlled test of whether the default LoRA rank (r=8) misses task-specific optima. We fine-tuned ~200 LoRA
adapters: Qwen2.5-0.5B on 6 tasks × 8 ranks (1–128) × 2 seeds, a single-seed 1.5B grid, and a rank × learning-rate
sweep. We also tested whether the spectrum of the full-weight gradient at initialization predicts the needed rank.
Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings
- **At the fixed recipe lr (2e-4, alpha=2r), the best rank spans r=1 to r=128, but in the reverse of the
  hypothesized order.** GSM8K and code do best at *low* rank and degrade at high rank (GSM8K 37.3% at r=1 vs 30.1%
  at r=128; code 28.8% at r=16 vs 22.7% at r=128). 6-way emotion classification is the only task that keeps improving
  up to r=32–128.
- **r=8 costs at most ~2.6 pp** (emotion: 0.5B +2.6 pp, cross-fitted +1.8 pp; 1.5B +2.2 pp). That is below the
  hypothesized 3–8 pp, and no generative task gains more than 1 pp (within noise) from any rank above 8.
- **The apparent rank need is mostly a learning-rate effect.** With lr tuned per rank, emotion is flat across ranks
  (r2 92.1%, r8 92.8%, r128 92.4%), code is flat (~28–30%), and GSM8K is best at r=8. The best lr shrinks as rank
  grows.
- **The gradient intrinsic dimensionality at init orders tasks consistently, but it anti-predicts the knee.** The
  ordering (code ≫ GSM8K > BoolQ > AG News > emotion ≈ SST-2) is the same for both model sizes, yet
  Spearman ρ = −0.74 vs the knee (p = 0.014, n = 12), and it lands within 2× of the knee for 0/12 pairs,
  vs 6/12 for "always r=8".
- **Model size (H3): weak evidence only.** The 1.5B knee is lower on 3/6 tasks, equal on 3 and never higher
  (sign test p = 0.25). 1.5B is much more robust to the r=128 collapse (BoolQ −2.5 vs −10.9 pp).

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate
uv add torch transformers peft datasets accelerate pandas scipy matplotlib pyarrow
export HF_HOME=/data/transformers HF_HUB_OFFLINE=1   # Qwen2.5-0.5B/1.5B + datasets from the local HF cache
# one run:
python src/train_eval.py --model 0.5B --task emotion --rank 8 --seed 0 --lr 2e-4
# gradient spectrum at init:
python src/grad_id.py --model 0.5B --task code
# full grid via the shared-GPU queue (job lists in results/jobs_*.txt; format: mem_gb<TAB>output<TAB>command):
python src/gpu_queue.py results/jobs_main.txt logs/jobs 0,1,2,3
# analysis + tables + figures:
python src/analyze.py && python src/make_tables.py
```
The code training set (`datasets/raw/code_alpaca_py.parquet`) comes from
`huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca`. The other datasets (SST-2, emotion, AG News,
BoolQ, GSM8K, MBPP) are read from the local HF hub cache; their paths are listed in `src/tasks.py`.

## Files
- `src/tasks.py`: data loading, prompt formats, answer extraction
- `src/train_eval.py`: one LoRA train + eval run (per-item results, adapter checkpoint)
- `src/code_exec.py`: MBPP pass@1 execution
- `src/grad_id.py`: gradient spectrum at initialization
- `src/gpu_queue.py`: memory-aware job queue for shared GPUs
- `src/analyze.py`, `src/make_tables.py`: statistics, figures and tables
- `results/runs/` (per-run JSON), `results/grad_id/`, `results/*.csv`, `results/tables.md`, `figures/`
- `planning.md`: pre-registered plan
