"""DOTS -- density-operator time-series statistics: the library behind DOMIC.

Module map:

    baselines  Gaussian likelihood-ratio baseline (Supplementary Table B.6)
    bmctc      BMCTC, our reimplementation of the matrix-information baseline of [4] (B.13)
    detect     feature density operators on the candidate grid; Holevo statistics of the ablation (B.2)
    domi       segment DOMI, the joint-state Holevo statistic and the dependence baselines on the
               same rank inputs
    encode     prefix-sum segment moments
    evaluate   power and localization summaries at a calibrated threshold
    extras     random Fourier features, studentization
    hourly     interval-aggregated segment statistics for long hourly series (Section 6.7)
    pelt       Holevo partitioning with the energy-weighted von Neumann cost (Section 4.4)
    perm       permutation p-value comparison with a numerical tie tolerance
    persist    per-replicate records written next to every aggregate
    qdiv       von Neumann and Shannon entropies from eigenvalues
    segtests   segmentation and single-break tests of the financial analyses (Section 6.8)
    synth      synthetic scenarios: article S1-S4 (codes D1-D4), M1, M2, MB (code P1); S1, S3 and
               N1-N3 of the ablation
    synth_ng   non-Gaussian dependence-change scenarios G1-G6
"""
