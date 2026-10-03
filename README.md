# Code and results for *DOMIC: Provably Calibrated Detection of Dependence-Structure Change Points via Density-Operator Mutual Information*

Jiwon Kang, Yun Am Seo — Jeju National University

Everything needed to reproduce the numbers, tables and figures of the article and its
supplementary material, and nothing else. All computation is classical. For a human-readable
digest of the headline results, see `RESULTS.md`.

    00_SRC/dots/     the library: density operators, entropies, DOMI, Holevo partitioning, baselines, scenarios
    00_SRC/*.py      one script per experiment or verification
    04_DAOU/         the outputs those scripts wrote — aggregates ship with per-replicate
                     records beside them (record files above 5 MB are omitted for size and
                     regenerate exactly from the fixed seeds)

Python 3.9+, NumPy, SciPy, Matplotlib. Every experiment is seed-deterministic. The runners use
multiprocessing, so pin the BLAS threads:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e1_matched_null.py --scen D1 --li 0

## Names in the code and in the article

| Article | Code and output files |
|---|---|
| DOMI (density-operator mutual information), random-feature form | `DOMI`, `DOMI-diff`, `dots/domi.py`, `primal-D8` in `scaling/` |
| Gram form of DOMI; its order-2 variant | `matmi-a1`; `matmi-r2`; `dual-full` in `scaling/` |
| DOMIC (the method) | the pipeline of `dots/domi.py`, `dots/pelt.py` and the runners |
| scenarios S1–S4, M1, MB; the marginal-shape control of B.15 | `D1`–`D4`, `M1`, `P1`; `M2` |
| non-Gaussian scenarios G1–G6 | `G1`–`G6` (`dots/synth_ng.py`) |
| raw and studentized variants of a statistic | modes `g` and `gs` |
| joint-state Holevo statistic (Section 4.2) | `Holevo-joint` |
| Holevo partitioning (Section 4.4) | `Holevo-partition` (method key in `e2`, `seg_baselines`, `mb_location`); the search itself is in `dots/pelt.py` |
| stage-two re-test (Section 4.7) | `retest` in script, file and key names (`run_e8_retest.py`, `retest_K999.json`); `passed` marks a candidate that passes it |
| random-date check (Section 6.7, B.8) | `randomdate` (a `--part` of `run_e8_season.py`, and JSON keys) |
| unrestricted stage two at K = 999 (B.8) | keys `*_unrestricted` and `unrestricted_candidates` in the `e8*` outputs |

## What produces what

Section numbers are the article's; B.x and A.x are sections of the Supplementary Material.

| Article item | Script | Output (`04_DAOU/EXPERIMENT/`) |
|---|---|---|
| Table 1, upper block; Figure 2 | `run_experiment.py --part e1`, `run_e1_matched_null.py` (one cell per call), `rebuild_table1.py`; Gram form `run_matmi_baseline.py` via `run_gram_full.sh`; copula test with ranks within each segment `run_cvm_subsample.py` (statistic in `cvm_subsample.py`); assembled by `build_tables.py` | `e1`, `e1_matched_null`, `matmi_baseline_full_T1a`–`T1e`, `cvm_subsample`, `tables/t1_combined.tex` |
| Table 1, lower block (non-Gaussian suite) | `run_ng_suite.py --tag A` (G1–G3), `--tag B` (G4–G6), Gram form `--tag gfull_Gk` (in `run_gram_full.sh`); `build_tables.py` | `ng_suite_A`, `ng_suite_B`, `ng_suite_gfull_G1`–`G6`, `tables` |
| Table 2 (cost of the two forms), B.9 | `run_scaling.py`, `build_tables.py` | `scaling`, `tables/t5_timing.tex` |
| Tables 3 and 4 (three-break benchmark; US financial panel) | `build_main_tables_mb_finance.py` (renders them from the outputs of §6.3 and §6.8) | `tables/table3_mb.tex`, `tables/table4_finance.tex` |
| §4.3 permutation nulls (pair vs Y-only); B.3 quantile against p-value rule | `run_perm_null_check.py` | `perm_null` |
| §4.3, Table B.5, B.17 block permutation | `run_block_perm_check.py` | `block_perm` |
| §4.3, B.17 block-permutation power | `run_block_power.py` | `block_power` |
| §4.3, §6.7, §6.8, B.17 exchangeability diagnostic | `run_exch_diag.py` | `exch_diag` |
| §4.5, B.18 combined statistic | `run_combo_check.py` | `theory/combo_check.csv` |
| §4.7 marginal-scale test | `run_marginal_validity.py` | `marginal_validity` |
| §6.2 Holevo partitioning under a marginal change | `run_pelt_specificity.py` | `pelt_specificity` |
| §6.3, Table B.7, Figure B.4 | `run_experiment.py --part e2`; `run_domi_binseg.py`; KCP, e.divisive and the re-test `run_seg_baselines.py` (methods in `seg_baselines.py`); location recovery of all three breaks from the stored segmentations `mb_location_recovery.py` | `e2`, `domi_binseg`, `seg_baselines`, `mb_location` |
| §6.4 copula shape at fixed Kendall's tau | `run_s3_mechanism.py` | `s3_mechanism` |
| §6.5, Table B.8 | `run_experiment.py --part e3`; per-replicate records of the S4 $r=0.85$ and M1 rows by `run_e3_cell.py` (one cell per call, same seeds) | `e3` |
| §6.7 weather, two-stage; Figure 3; B.8 | `extract_weather_hourly.py`, `prep_weather_hourly.py`, then `run_e8_hourly.py` (stage one), `run_e8_season.py` (season-restricted stage two, random-date check, split sample), `run_e8_validate.py` (unrestricted random-date check), `run_e8_split.py`, `run_e8_feature_draws.py`; unrestricted stage two `run_e8_retest.py` (B.8) and its K = 9999 extension `run_e8_retest_hiK.py` (read by B.14); shared routines `run_e8_marginal.py` | `e8`, `e8_season`, `e8_validate`, `e8_split` |
| §6.7, B.8 effective sample sizes (q-values of §6.7 come from `run_e8_season.py`; `run_e8_summary.py` supplies the step-up routine) | `run_e8_ess.py`; `run_e8_summary.py` | `e8/effective_sample_sizes.json`, `e8/retest_summary.json` |
| §6.7 block exchangeability of the records | `run_block_exch_diag.py` | `block_exch_diag` |
| §6.8 finance, Figure B.5; B.8 | `run_e6_finance.py`, `run_e6a_block.py`, `run_e6_monthly.py`, `run_block_exch_diag_fin.py`; stage two `run_e6_retest.py` | `e6`, `block_exch_diag` |
| Table B.6 (joint-distribution references) | as Table 1, upper block; `build_tables.py` | `e1_matched_null`, `tables/tb_jointref.tex` |
| B.1 sensitivity (also quoted in B.3 and B.17) | `run_experiment.py --part e4` | `e4` |
| B.2 ablation, Figure B.1; paired McNemar comparison | `run_ablation.py`; `run_ablation_paired.py` | `../ABLATION` |
| B.3, Figure B.2; A.6, A.7, A.10; §6.2 marginal shifts on a fixed dependence | `run_theory_checks.py`, `run_theory_extras.py` | `theory`, `theory_extras` |
| §6.1, B.4 model-based baselines | `run_e7_modelbased.py` | `e7` (S4: see next row) |
| B.4: the same baselines on S4 with each level calibrated against its own null (the S4 null depends on r) | `run_e7_matched_d4.py` (re-scores the stored `e7` alternatives) | `e7_matched_d4` |
| §6.6, B.5 feature draw, break location, calibration draw | `run_design_checks.py` (needs `extract_weather.py`), `check_calibration_draw_effect.py` | `design_checks`, `diagnostics` |
| B.6 vector blocks, Table B.1; population contrast | `run_mv_blocks.py`, `run_mv_contrast.py` | `mv_blocks` |
| B.7, Table B.2 | `run_bandwidth_matched_null.py` + `rebuild_tableB2.py`, `run_hsic_fullgram.py`, `run_domi_bandwidth.py`, `run_domi_narrow_D.py` | `bandwidth_matched_null`, `hsic_fullgram`, `domi_bandwidth`, `domi_narrow_D` |
| B.10, Table B.3 (Gram form, feature dimension); subsample comparison | `run_matmi_baseline.py` (runs listed in its docstring), `run_gram_subsample_check.py`, `build_tables.py` | `matmi_baseline*`, `gram_subsample_check`, `tables/tb10_gram.tex` |
| B.11, Table B.4 | `build_tables.py` | `tables/tb11_nongauss.tex` |
| [NV-1] condition (A3) for S2 (A.7); [NV-4] bias constant, closed form and measured (A.10) | `proof_numbers.py` | `theory/proof_numbers.json` |
| [NV-5]–[NV-7] simulation studies | `run_theory_sims.py` | `theory_sims` |
| [NV-2] (A.8): deployed feature map on the Frank copula, grid refinement | `check_nv2_grid.py` | `theory/nv2_grid.json` |
| A.9, A.10: eigenvalue floor, smallest eigenvalues, marginal entropies (also §4.1) | `run_spectrum_facts.py` | `spectrum_facts` |
| §4.3 level of the adaptive calibration protocol | `run_protocol_level.py` | `protocol_level` |
| §6.2, B.15 marginal-shape control for S2 (scenario M2) | `run_marginal_shape_control.py` | `marginal_shape_control` |
| B.8 coverage of the hourly weather record | `weather_coverage.py` (needs the weather data) | `e8/data_coverage.json` |
| A.5, B.8 tied observations and the permutation level; weather stage two with ties broken at random | `run_tie_check.py`, `weather_tie_shares.py` (needs the weather data); `run_e8_season.py` with `E8_TIES=random` (parts cand, randomdate, split, summary) | `tie_check`, `e8_season_randomties` |
| §4.6, B.16, Table B.13: calibration by the boundary limit law | `run_asymptotic_threshold.py` (method in `asymptotic_threshold.py`); share of null maxima near the grid ends `summarise_asymptotic_ends.py` | `asymptotic_threshold` |
| §4.8, B.9: order-2 Gram form by prefix sums; end-to-end cost | `gram_r2_prefix.py`, `run_endtoend_cost.py` | `gram_r2_prefix`, `endtoend_cost` |
| B.13, Table B.9: BMCTC (our reimplementation of [4]) | `run_bmctc.py` (method in `dots/bmctc.py`) | `bmctc` |
| B.14: permutation test of the whole two-stage procedure | `run_e8_fullperm.py`, `run_fullperm_synth.py` | `e8_fullperm`, `fullperm_synth` |
| B.15, Tables B.10–B.12: detection rates, pair-permutation comparison, off-center breaks | `build_detect_rates.py` (its copula column is the global variant; Table B.10 takes the copula column from `cvm_subsample`), `run_perm_compare.py` (Gram form on GPU or CPU by `gram_gpu.py`), `run_offcentre.py`; tables by `build_tables.py` | `detect_rates`, `perm_compare`, `offcentre`, `tables` |
| B.19, Table B.14: simulation on empirical weather margins (needs the weather data below) | `run_databased_sim.py` | `databased_sim` |
| B.20, Table B.15: MC-TIRE [17] under the calibration of Table 1 | `run_mctire.py` (the authors' code through `mctire_adapter.py`; see below) | `mctire` |
| B.9, Table B.16: test-computation time of every statistic, from ranks through the p-value (wall-clock times depend on the machine) | `run_cost_compare.py` | `cost_compare` |
| B.8: level at a 91% tie share, 2000 further replicates | `run_tie_check_extreme.py` (the test of `run_tie_check.py`) | `tie_check_extreme` |
| B.18: combined statistic on the 2021-2022 stock-bond window under pair and block permutation (one permutation draw) | `run_combined_block.py` | `combined_block` |
| Section 6.8, B.4, B.18, Table B.19: the same window over 20 independent permutation draws of K = 999 | `run_combined_block_seeds.py` (the test of `run_combined_block.py`) | `combined_block_seeds20` |
| B.21, Table B.17: sign reversal of a Gaussian correlation | `run_signflip.py` | `signflip` |
| B.13, Table B.18: BMCTC with its input and kernel rule varied one at a time | `run_bmctc_factorial.py` | `bmctc_factorial` |
| B.6, Table B.1: margins-only control MV-M1 at d = 1 over four further draws of replicates and null calibration | `run_mv_m1_draws.py` | `mv_m1_draws` |
| B.22: exploratory whole-record scans of the financial pairs that preceded the protocol (K = 199) | `run_e6_fullscan.py` | `e6_fullscan` |
| Section 6.8, B.22, Tables B.20-B.21: US financial panel (S&P 500 against gold, the 10-year yield and the dollar index): whole-record tests, stage two and descriptive statistics | `prep_us_finance.py`, `run_us_finance_panel.py`, `run_us_finance_stage2.py`, `describe_us_panel_breaks.py` | `us_panel` |
| [NV-3] (A.8): norm range of the deployed feature map on a grid | `check_rank_correction_algebra.py` | `theory_rank_process/rank_correction_algebra.json` |
| [NV-8] (B.3, cited in A.12): global against within-segment ranks at the boundary | `check_rank_process_boundary.py` | `theory_rank_process` |
| Figure 1 | `make_fig1.py` | `figures/` |
| Figure 2, Figures B.1–B.5 | `make_paper_figures.py` | `figures/` (created on first run) |
| Figure 3 (needs the weather data below) | `FIGURE3_LAYOUT=column python 00_SRC/make_figure3_weather.py` (the printed single-column layout; without it the script adds a panel of the class counts given in Section 6.7) | `figures/` |

MC-TIRE runs in a separate environment with TensorFlow 2.15 and TensorFlow Probability 0.23. Clone the
authors' repository, https://github.com/caozhenxiang/MC-TIRE, at commit f538a86 into `00_SRC/_ext/MC-TIRE`;
it is used unchanged. `python 00_SRC/run_mctire.py --prepare` writes the input series from the article's
generators (`inputs_sha256.json` records their hashes), the TensorFlow environment runs
`run_mctire.py --run`, and `python 00_SRC/run_mctire.py --evaluate` calibrates the curves. The per-replicate
curves (82 MB) are not shipped; `records_null.csv.gz` and `records_alt.csv.gz` hold each replicate's maxima
and break estimates.

Figure files are named after the figure they print (`figure_2` is Figure 2, `figure_B1` Figure B.1,
and so on). `build_tables.py` writes one row per printed number to
`tables/provenance.csv` (source file, scenario, level, statistic, variant, value).

In `theory_sims/records_nv5_reps.csv` and `records_nv6_maxima.csv` the `rep` column is a running index over all jobs of the run, not the replicate seed; rows are in replicate order within each configuration.

`rebuild_table1.py` and `rebuild_tableB2.py`
assemble Table 1 and Table B.2 from the stored aggregates and write them as
`e1_matched_null/table1.csv` and `bandwidth_matched_null/tableB2.csv`; `table1.csv` holds the
single-break numbers one row per statistic, which the article prints as Table 1 (one row per
setting) and Supplementary Table B.6. The per-replicate null records behind `e1_matched_null/results.csv` exceed the size cap
and are not shipped — the fixed seeds of `run_e1_matched_null.py` regenerate them bit-identically,
and `check_calibration_draw_effect.py` (B.5) needs them regenerated first. `e1/results.csv` ships as an aggregate only,
without per-replicate records (it backs the cells
of Table 1 that do not depend on the null level; the others carry full records in `e1_matched_null`).

Every permutation p-value is (1 + #{k : T_k >= T_0}) / (K + 1), where a replica counts as T_k >= T_0 when
T_k >= T_0 - 1e-9 max(1, |T_0|) (`dots/perm.py`): different permutations can give equal statistics, and
rounding can place such a replica marginally below T_0. The p-values are computed before the statistics are
written to the record files; the stored statistic vectors are rounded and describe the
permutation distributions, but they do not preserve every tie comparison.

The [NV-k] rows produce the computed numbers that statements in the Supplementary Material rest on,
where the proofs are given; each script recomputes them from fixed seeds.

## Data

The observations are third-party and are not redistributed; both sources are openly accessible
and the scripts rebuild the analyzed series from them.

* **Weather.** Hourly surface (ASOS) observations at twelve stations, 2018–2025, from the Korea
  Meteorological Administration API Hub, <https://apihub.kma.go.kr> (free registration; service
  `kma_sfctm2.php`, one request per hour with `stn=0` for all stations, or `kma_sfctm3.php` for a
  range). Save one CSV per day as `<YYYY>/<MM>/asos_YYYYMMDD.csv` with the columns the service returns
  (`tm`, `stn`, `ta`, `wd`, `ws`, `td`, `hm`, `pa`, ...). Some hours and values are absent from the
  service; `weather_coverage.py` reports what the analyses use (B.8). Set the environment
  variable `ASOS_ARCHIVE` to the download directory; `extract_weather_hourly.py` reads every
  file matching `$ASOS_ARCHIVE/*/*/asos_*.csv` (any two-level layout), keeps CSV columns
  `tm` (timestamp `YYYYMMDDHHMM`), `stn` (station id), `ta`, `hm`, `ws`, `pa`, `td`, and writes
  `02_MART/WEATHER_HOURLY.csv`; then run `prep_weather_hourly.py` to build the anomaly series.
  `extract_weather.py` builds the daily file `02_MART/WEATHER_DAILY.csv` from the same archive.
* **Finance.** Daily closing levels, 2011–2026, from Yahoo Finance — S&P 500 `^GSPC`, CBOE
  10-year Treasury yield `^TNX`, KOSPI `^KS11`, USD/KRW `KRW=X`, SPDR Gold Shares `GLD`, US Dollar
  Index `DX-Y.NYB`. Save them as `01_ORG/FINANCE/yahoo_{gspc,tnx,kospi,usdkrw,gld,dxy}.csv` with
  header `date,close`, dates as `YYYYMMDD` strings (e.g. `20210104`), one row per trading day. For
  `GLD` and `DX-Y.NYB` the daily chart response
  `https://query1.finance.yahoo.com/v8/finance/chart/<ticker>?period1=1313971200&period2=1787270400&interval=1d`,
  saved as `01_ORG/FINANCE/raw_{GLD,DXY}.json`, is converted to these files by `prep_us_finance.py`.
  Closes revised by the provider after our download can differ; `us_panel/config.json` records the
  SHA-256 of the input files the reported results were computed from, and `run_us_finance_panel.py`
  stops if the inputs differ unless `US_PANEL_OUT` names a new output directory.

Licensed under the MIT License (see `LICENSE`).

## Citation

If you use this code or results, please cite the arXiv preprint:

    @misc{kang2026domic,
      title   = {{DOMIC}: Provably Calibrated Detection of Dependence-Structure Change Points
                 via Density-Operator Mutual Information},
      author  = {Kang, Jiwon and Seo, Yun Am},
      year    = {2026},
      eprint  = {2609.02787},
      archivePrefix = {arXiv},
      primaryClass  = {stat.ME},
      url     = {https://arxiv.org/abs/2609.02787}
    }
