# Results digest

To check a number in the article: find its item in the README table (script and
output directory), or start from the digests below, which are computed from the shipped
output files -- every number here is also in those files, with per-replicate or
per-permutation records beside them where the README does not note otherwise.
Paths below are relative to `04_DAOU/EXPERIMENT/`. Article tables: `tables/*.tex`
(built by `build_tables.py`; Table 1 and B.2 inputs in `e1_matched_null/table1.csv` and `bandwidth_matched_null/tableB2.csv`).

## Weather two-stage analysis (Section 6.7; season-restricted stage two, K = 9999)

Stage-two p-values are nominal: stage one selected the dates from the same data.

| station | pair | date | dependence p | BH q | BY q | marginal p (X / Y) | class |
|---|---|---|---|---|---|---|---|
| Suwon | hm-ws | 20190826 | 0.0005 | 0.013 | 0.052 | 0.016 / 0.042 | both change |
| Jeonju | hm-ws | 20191021 | 0.0179 | 0.200 | 0.777 | 0.088 / 0.090 | undetermined |
| Daegu | ta-hm | 20190715 | 0.0237 | 0.200 | 0.777 | 0.981 / 0.887 | undetermined |
| Seoul | ta-ws | 20190909 | 0.0335 | 0.200 | 0.777 | 0.409 / 0.000 | marginal-driven |
| Jeonju | ta-hm | 20191021 | 0.0370 | 0.200 | 0.777 | 0.264 / 0.086 | undetermined |
| Jeju | ta-hm | 20230612 | 0.1079 | 0.486 | 1.000 | 0.446 / 0.467 | undetermined |
| Suwon | hm-ws | 20200921 | 0.1292 | 0.498 | 1.000 | 0.533 / 0.416 | undetermined |
| Incheon | ta-hm | 20230515 | 0.1694 | 0.572 | 1.000 | 0.942 / 0.980 | undetermined |
| Incheon | ta-ws | 20221115 | 0.2516 | 0.755 | 1.000 | 0.347 / 0.320 | undetermined |
| Seoul | hm-ws | 20200406 | 0.2857 | 0.771 | 1.000 | 0.129 / 0.041 | marginal-driven |
| Mokpo | ta-hm | 20180522 | 0.3152 | 0.774 | 1.000 | 0.064 / 0.524 | undetermined |
| Daegu | hm-ws | 20190603 | 0.3652 | 0.791 | 1.000 | 0.810 / 0.005 | marginal-driven |
| Jeju | ta-ws | 20210111 | 0.4053 | 0.791 | 1.000 | 0.112 / 0.282 | undetermined |
| Gangneung | hm-ws | 20190603 | 0.4221 | 0.791 | 1.000 | 0.947 / 0.527 | undetermined |
| Gangneung | ta-ws | 20180716 | 0.4396 | 0.791 | 1.000 | 0.105 / 0.353 | undetermined |
| Daejeon | ta-ws | 20211213 | 0.5297 | 0.894 | 1.000 | 0.171 / 0.720 | undetermined |
| Gwangju | ta-ws | 20210111 | 0.5725 | 0.900 | 1.000 | 0.239 / 0.476 | undetermined |
| Gwangju | hm-ws | 20240514 | 0.6250 | 0.900 | 1.000 | 0.322 / 0.370 | undetermined |
| Daejeon | hm-ws | 20211227 | 0.7184 | 0.900 | 1.000 | 0.488 / 0.653 | undetermined |
| Mokpo | hm-ws | 20230404 | 0.7755 | 0.900 | 1.000 | 0.512 / 0.662 | undetermined |
| Seoul | ta-hm | 20200420 | 0.7798 | 0.900 | 1.000 | 0.103 / 0.432 | undetermined |
| Incheon | hm-ws | 20220613 | 0.7805 | 0.900 | 1.000 | 0.282 / 0.254 | undetermined |
| Mokpo | hm-ws | 20180521 | 0.7836 | 0.900 | 1.000 | 0.519 / 0.443 | undetermined |
| Mokpo | ta-hm | 20231017 | 0.8003 | 0.900 | 1.000 | 0.413 / 0.039 | marginal-driven |
| Jeju | hm-ws | 20230612 | 0.9314 | 0.982 | 1.000 | 0.463 / 0.457 | undetermined |
| Gwangju | hm-ws | 20230515 | 0.9521 | 0.982 | 1.000 | 0.780 / 0.260 | undetermined |
| Jeonju | ta-ws | 20250609 | 0.9821 | 0.982 | 1.000 | 0.187 / 0.547 | undetermined |

Random-date check (720 random dates): dependence test rejects at 5.4% (0.05) and 0.42% (0.01); marginal test (smaller of two p-values) at 11.4% (0.05). The unrestricted scheme is in `e8_validate/results.json`.


## Finance segmentations (Section 6.8; block-calibrated)

| pair | weekly breaks (block permutation) | monthly breaks |
|---|---|---|
| SPX-TNX | 20200227 | 20210830 (pair) |
| KOSPI-USDKRW | not calibrated (spurious rate 0.60 at the largest penalty) | 20180110 (block12) |

## Finance stage two (Section 6.8)

| break | dependence p | marginal p (X / Y) | class | draws with p <= 0.05 |
|---|---|---|---|---|
| SPX-TNX weekly 2020-02-27 | 0.054 | 0.251 / 0.216 | undetermined | 40% |
| SPX-TNX monthly 2021-08 | 0.988 | 0.002 / 0.041 | marginal-driven | 0% |
| KOSPI-USDKRW monthly 2018-01 | 0.395 | 0.010 / 0.600 | marginal-driven | 0% |

## Daily 2021-2022 stock-bond window (Section 6.8, Supplementary Table B.19)

Median (range) of the single-break p-value over 20 independent permutation draws of K = 999 each, all on
the same observed window: the spread is Monte Carlo variation, not variation over independent data.

| statistic | pair permutation | block b=20 | block b=50 |
|---|---|---|---|
| DOMI-diff | 0.268 (0.223-0.303) | 0.231 (0.202-0.272) | 0.3605 (0.298-0.408) |
| combined (Section 4.5) | 0.016 (0.008-0.022) | 0.1475 (0.095-0.183) | 0.4335 (0.358-0.493) |
| Spearman-diff | 0.006 (0.003-0.010) | 0.057 (0.035-0.078) | 0.2965 (0.241-0.340) |
| HSIC-diff | 0.026 (0.017-0.049) | 0.0195 (0.009-0.033) | 0.078 (0.050-0.105) |
| copula CvM, global ranks | 0.001 (0.001-0.002) | 0.001 (0.001-0.001) | 0.002 (0.001-0.003) |

The Section 4.3 diagnostic rules out pair permutation for this window; it is shown for comparison.
The copula statistic from global ranks also responds to changes confined to the margins (Supplementary
Table B.11).

## In-text rates quoted in Sections 4.3 and 6.2

| claim in the article | value | file |
|---|---|---|
| Y-only permutation, constant Gaussian-copula null (Sec 4.3) | 96.5% | `perm_null/results.json` |
| Y-only permutation, constant nonlinear null | 26.5% | `perm_null/results.json` |
| pair permutation on the same two nulls | 7.0% and 6.0% | `perm_null/results.json` |
| Y-only and pair permutation on the independent null | 6.0% and 5.0% | `perm_null/results.json` |
| quantile rule vs p-value rule at K=99 (Sec 4.3) | 7.0% vs 5.0% | `perm_null/results.json` |
| Holevo partitioning segments a pure marginal change (Sec 6.2), s=1.3/1.6/2.0 | 0.45 / 1.00 / 1.00 (null rate 0.04) | `pelt_specificity/results.json` |

Block-permutation level and power (Supplementary Table B.5, Sec 4.3): `block_perm/results.csv`,
`block_power/results.csv`. Aggregates sit next to their per-replicate or per-permutation
records where the README does not note otherwise; the README table maps every article item to the script
that produced it and the directory it lives in.
