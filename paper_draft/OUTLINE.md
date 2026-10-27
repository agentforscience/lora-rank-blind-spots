# Outline: The Default LoRA Rank Is Not the Blind Spot

1. Intro: hook (r=8 hard-coded) -> claims that code/math need high rank -> gap (no controlled sweep w/ lr control, no cheap predictor) -> our sweep (~200 adapters, 6 tasks x 8 ranks x 2 models, lr sweep, grad-ID) -> preview numbers (reversed ordering; emotion +2.6pp gap vanishes w/ tuned lr; rho=-0.74) -> contributions.
2. Related work: LoRA & rank choice (Hu, AdaLoRA); intrinsic dimension (Aghajanyan, Roy&Vetterli); rank-lr coupling (rsLoRA); high-rank for code/math (Biderman).
3. Method: problem formulation (knee, best rank, gap), tasks table, training recipe, lr sweep, grad-ID predictor, statistics.
4. Results: rank curves (Tab main, Fig1), r8 gap (Tab gap), lr sweep (Tab lr, Fig2), model size (Tab size), predictor (Tab gradid, Tab corr, Fig3/4).
5. Discussion: verdicts table, why math/code low-rank, why predictor fails, failure modes, limitations, implications.
6. Conclusion + future work.
