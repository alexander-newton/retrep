# 184: Finkelstein, Notowidigdo, Schilbach & Zhang (2025)

*"Lives vs. Livelihoods: The Impact of the Great Recession on Mortality and Welfare."*
Private: the data are built from CDC WONDER and are not for distribution.

## Layout (as for the other retrep papers)

| Path | Contents |
|---|---|
| `code/184/184_replication_<N>.py` | one self-contained script per replication, N = 1–52 |
| `rawdata/184/184_<N>_<name>.dta` | the input data each script reads (`config['rawdata']`) |
| `output/184/184_<N>/` | `y.parquet`, `X.parquet`, `metadata.json` written by script N (pre-built) |

Each script reads its `.dta`, builds `y` and `X`, writes them with the metadata to
`output/184/184_<N>/`, and calls `replicate()`. The pre-built `output/184/` was produced by
running these scripts unchanged. `replicate()`'s own `result_table_*` folders are not
included, because `replicate()` could not be run where the package was built.

**What the estimator needs:** `y.parquet`, `X.parquet` and `metadata.json` hold all the
data for a run. As with the other papers, the role of each column comes from the
convention, not from `metadata.json`:
- **interest** = first column of `X` (`shock_x_post`);
- **fixed effects** = the `yr_*` dummies and `cz`;
- **weight** = `czpop_2006`, the last column (not present in 184_12);
- **cluster** = `cz`.

## What every script does

The scripts are self-contained and identical except for the data file and the header.
The regression is the authors' main event study (`NCHS_mortality_analysis.do`, `run_fereg`):

```
areg ln_age_adj_mort_plus1 ib2006.year##c.shock [aweight=czpop_2006], absorb(cz) cluster(cz)
```

It covers 741 commuting zones (CZs) × 2003–2016, with CZ and year fixed effects,
weights = 2006 CZ population, clustered by CZ. The paper's headline number is the
**average of the 2007–2016 year × shock coefficients** (×100: −0.558, SE 0.279).

The retrep convention is a single variable of interest in the first column of `X`, so
each script uses an **exact reparametrisation** of that regression:

| `X` column | Definition |
|---|---|
| `shock_x_post` (**interest**) | shock × 1(year ≥ 2007) |
| `shock_x_2008_minus_2007` … `shock_x_2016_minus_2007` | shock × 1(year = t) − shock × 1(year = 2007) |
| `shock_x_2003` … `shock_x_2005` | shock × 1(year = t): pre-period placebos (base 2006) |
| `yr_2004` … `yr_2016` | year fixed effects (`get_dummies`, first year dropped) |
| `cz` | CZ fixed effect (absorbs the shock main effect) |
| `czpop_2006` | weight, last column (like `freq` in paper 183); not in 184_12 |

The coefficient on `shock_x_post` equals the average of the authors' 2007–2016
coefficients, and the fit, residuals and standard errors are the same. This was verified
numerically: −0.5417 (SE 0.2771) both ways.

**y** = `age_adj_mort_plus1`, age-adjusted mortality per 100k + 1, in levels;
`log(y)` is the authors' DV. **model_type** = `log-linear`.

`replicate()` is called with `X[['shock_x_post'] + controls]`, `fe = X[year_cols + ['cz']]`
and CZ-clustered errors via `kwargs_ols` (as in `code/004`). Note that `replicate()`
currently fits OLS without weights, so its OLS output will look like 184_12, not 184_1.

## Numbering

| Replication | Analysis | Replication | Analysis |
|---|---|---|---|
| 184_1 | main (preferred) | 184_7 | tiny county-years at 9 |
| 184_2 | age split by state rates | 184_8 | CZs with imputed share ≤ 20% |
| 184_3 | age split uniform | 184_9 | … ≤ 10% |
| 184_4 | highest possible rates | 184_10 | … ≤ 5% |
| 184_5 | lowest possible rates | 184_11 | without CZs with a CZ-year < 20 deaths |
| 184_6 | tiny county-years at 0 | 184_12 | unweighted |
| | | 184_13 – 184_52 | multiple-imputation draws 1–40 |

## The analyses and how closely they match

The 2007–16 average ×100 is from weighted OLS on each script's own `y`/`X` (184_12:
unweighted); the published figure is −0.558 (SE 0.279).

| Script | Analysis | N (CZs) | 2007–16 (SE) | How closely it matches the paper |
|---|---|---|---|---|
| **184_1** | **Main build**: hidden cells (0–9 deaths) rebuilt from exact county-year totals; age split by national death rates | 10,374 (741) | **−0.542 (0.277)** | **Closest; the one to report.** Same sample, weights and clustering. Gap −0.016 = 6% of an SE; all 13 year coefficients within 0.0002 of published. |
| 184_2 | Age split by state death rates | 10,374 (741) | −0.541 (0.277) | Same as 184_1. |
| 184_3 | Age split even across hidden cells | 10,374 (741) | −0.566 (0.280) | Very close, despite a deliberately crude split. |
| 184_4 | Every CZ-year at its highest possible rate | 10,374 (741) | −0.570 (0.282) | Close. Bounds each CZ-year's outcome, **not** the coefficient. |
| 184_5 | Every CZ-year at its lowest possible rate | 10,374 (741) | −0.563 (0.276) | Close; same caveat as 184_4. |
| 184_6 | 502 county-years missing from WONDER (0–9 deaths) set to 0 | 10,374 (741) | −0.560 (0.278) | Closest numerically, but for the wrong reason: it creates 36 zero-mortality CZ-years. Use 184_1. |
| 184_7 | Same county-years set to 9 | 10,374 (741) | −0.541 (0.277) | Same as 184_1. |
| 184_8 | CZs with mean imputed share ≤ 20% | 6,958 (497) | −0.519 (0.292) | Different sample (98% of the weight); compare with 184_1. |
| 184_9 | … ≤ 10% | 4,634 (331) | −0.490 (0.306) | Different sample (92% of the weight). |
| 184_10 | … ≤ 5% | 2,478 (177) | −0.476 (0.361) | Different sample (80% of the weight). |
| 184_11 | Without the 13 CZs that have a CZ-year with < 20 deaths | 10,192 (728) | −0.541 (0.277) | Same as 184_1. |
| 184_12 | 184_1 without population weights | 10,374 (741) | −0.273 (0.148) | **Not comparable:** a different estimand (the average CZ, not the average person). |
| 184_13 – 184_52 | 40 random age splits and tiny-county totals | 10,374 (741) each | −0.548 (0.278) after Rubin's rules | Same sample as 184_1. Draws range from −0.562 to −0.538, so imputation adds almost no uncertainty. |

**Reading the table:**
- **184_1 – 184_7 and 184_11** lie between −0.54 and −0.57. The county-year totals are exact, so only
  the split of hidden deaths across ages is imputed, and that choice barely matters.
- **184_8 – 184_10** move toward −0.48 as small CZs are dropped. This mixes imputation with real
  differences between small and large CZs: the 100 largest CZs alone give −0.58.
- **184_12** differs because weighting changes the question.
- **184_13 – 184_52:** run your estimator on all 40 draws. The combined estimate is the mean over
  draws; the variance is the mean within-draw variance + (1 + 1/40) × the between-draw
  variance.

## Where the data come from

The `.dta` files are built by `wonder_pipeline/code/make_finkelstein_data.py` in the
Finkelstein replication-package folder (not part of retrep). The build is documented in
`wonder_pipeline/README.md`: public CDC WONDER county × age deaths, aggregated to CZs with
the authors' crosswalk, SEER populations, direct age adjustment to the US 2000 standard
exactly as in the authors' Stata build, and the Yagan (2019) shock. Running the scripts needs only the `.dta` files.
