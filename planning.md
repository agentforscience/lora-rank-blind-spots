# Planning: LoRA Rank Selection Blind Spots

## Motivation & Novelty Assessment

### Why This Research Matters
LoRA is the default way to adapt LLMs, and nearly every recipe hard-codes a rank (r=8 or 16) without justification.
If the best rank really varies by an order of magnitude across tasks, practitioners are systematically leaving
accuracy on the table for "hard" generative tasks (math, code) or wasting parameters/memory on easy ones. A cheap,
pre-training predictor of the needed rank (e.g. from gradients at initialization) would remove a costly sweep.

### Gap in Existing Work
(No literature review was pre-gathered for this workspace; the following is from background knowledge.)
- Hu et al. 2021 (LoRA) reported that very low ranks (r=1–8) suffice on GPT-3 for NLU/NLG benchmarks and that ΔW has
  very low "intrinsic rank"; Aghajanyan et al. 2020 showed low intrinsic dimensionality of fine-tuning, decreasing with
  model size.
- Later work (AdaLoRA, Zhang et al. 2023; rsLoRA, Kalajdzievski 2023; "LoRA Learns Less and Forgets Less",
  Biderman et al. 2024) found that code / math continued-training needs much higher rank, and that the standard
  alpha/r scaling makes high ranks under-perform unless learning rate / scaling is adjusted.
- What is missing: a *controlled* sweep that (a) holds data budget, optimizer and evaluation fixed across several task
  types and model sizes, (b) quantifies the cost of the r=8 default per task with uncertainty, and (c) tests whether a
  cheap, label-free-of-training statistic (spectrum of the full-weight gradient at init) predicts the knee rank.

### Our Novel Contribution
A controlled rank sweep (r = 1…128, 8 values) on 6 tasks spanning classification → math → code, on two model sizes of
one family (Qwen2.5-0.5B/1.5B base), plus a gradient-spectrum predictor evaluated by Spearman correlation against the
empirically determined knee rank.

### Experiment Justification
- Exp 1 (rank sweep, 0.5B, 6 tasks × 8 ranks × 2 seeds): needed to measure per-task optimal/knee rank and the rank-8 gap (H1, H2).
- Exp 2 (rank sweep, 1.5B, same grid): needed to test whether the knee shifts with model size (H3) and to add
  data points for the predictor test.
- Exp 3 (gradient intrinsic dimensionality at init per task × model): the proposed predictor (H4).
- Exp 4 (learning-rate sensitivity at selected ranks): a rank comparison at a single lr is confounded by the fact that
  the optimal lr depends on rank under alpha=2r scaling; we check whether conclusions survive per-rank lr tuning.

## Research Question
Does the optimal LoRA rank vary widely across tasks, how much does the common default (r=8) cost, does the knee rank
shrink with model size, and can the effective rank of the initial full-weight gradient predict it?

## Hypothesis Decomposition
- H1: max/min knee rank across tasks (per model) ≥ 8.
- H2: for the "high-rank" tasks, metric(best rank) − metric(r=8) is 3–8 pp (bootstrap CI excludes 0).
- H3: knee(1.5B) < knee(0.5B) on the same task for most tasks (paired, sign test).
- H4: Spearman(grad-ID, knee rank) > 0.7 over (task, model) pairs; a single calibrated scale puts the prediction
  within a factor 2 of the knee for most pairs.

## Proposed Methodology

### Approach
Base models: Qwen2.5-0.5B and Qwen2.5-1.5B (base, same family/tokenizer → clean size comparison; both fit the ~4–6 GB
GPU slices available on this shared host). 3B only if a GPU frees up.

Tasks (all converted to causal-LM SFT with loss only on the response tokens; 4,000 training examples each, 1 epoch):
| task | train | eval | metric |
|---|---|---|---|
| SST-2 sentiment | GLUE train | validation (872) | accuracy (label-restricted scoring) |
| Emotion (6-way) | dair-ai/emotion train | test (1,000 sample) | accuracy |
| AG News (4-way) | train | test (1,000 sample) | accuracy |
| BoolQ | train | validation (1,000 sample) | accuracy |
| GSM8K | train | test (all 1,319) | exact-match of final answer (greedy) |
| Python code | python_code_instructions_18k | MBPP test (500) | pass@1 by executing the 3 tests |
Every run also reports held-out target-token NLL (300 held-out training-distribution examples), a lower-noise
secondary metric.

LoRA config: all linear layers (q,k,v,o,gate,up,down), alpha = 2r (constant scale 2, the "r=8, alpha=16" convention),
dropout 0.05, AdamW lr 2e-4, cosine with 3% warmup, effective batch 16, max length 512, bf16 autocast.
Ranks: 1, 2, 4, 8, 16, 32, 64, 128. Seeds: 2 (0.5B), ≥1 (1.5B; second seed if time). Zero-shot (r=0) baseline.

### Definitions (pre-registered)
- Best rank: argmax of seed-averaged metric.
- Knee rank: smallest r whose seed-mean metric ≥ best − δ, δ = max(0.5 pp, 1 pooled SE of the metric between seeds/
  sampling). Also reported with δ = 1 pp, and on NLL (smallest r within 1% of the best relative NLL reduction gain).
- Rank-8 gap: best − r8, paired bootstrap over eval items (and seeds).
- Gradient ID: for every target matrix, accumulate the full-weight gradient of the SFT loss at the pretrained weights
  over 128 training examples; compute the singular-value spectrum; r90 = #singular values capturing 90% of the
  squared Frobenius norm. Predictor = median r90 over matrices (secondary: entropy effective rank, r50, r99,
  stable rank, top-8 energy fraction).

### Baselines
Zero-shot base model; the r=8 default; best-rank oracle. Predictor baselines: constant prediction (r=8) and the
zero-shot error rate of the task as a naive "difficulty" predictor.

### Statistical Analysis Plan
Paired bootstrap (10k resamples over eval items, seeds pooled) for gaps; Spearman with permutation p-value for H4;
sign test for H3; α = 0.05, Holm correction across the 6 tasks for the gap tests.

## Expected Outcomes
Supportive: classification knees at r≤4, GSM8K/code knees ≥32, rank-8 gap 3–8 pp on GSM8K/code, smaller knees for
1.5B, ρ > 0.7. Refuting: flat curves (all knees ≤8), or gaps within noise, or ρ ≤ 0.

## Timeline and Milestones
Setup + code 1 h; 0.5B sweep ~2 h on 3–4 slices in parallel; 1.5B sweep ~2–3 h; grad-ID 20 min; analysis + report 1 h.

## Potential Challenges
- Shared GPUs (~4–6 GB free): use a work queue that launches jobs only when free memory fits; gradient checkpointing
  for 1.5B; per-output flock to avoid duplicate writers.
- Noise between neighbouring ranks larger than effects → knee tolerance, 2 seeds, NLL secondary metric.
- Fixed-lr confound → Exp 4.

## Success Criteria
All four sub-hypotheses receive a clear supported / not supported verdict with CIs, documented in REPORT.md.
