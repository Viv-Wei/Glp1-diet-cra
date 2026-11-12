# Code and data

***Glucagon-like peptide-1 receptor agonists versus dietary policy for preventing cardiovascular deaths in 19 countries: a modelling study***

## Model

A comparative risk assessment and multistate life table follows adults aged 25 years or older in 19 countries and globally for 30 years (2024 to 2053), by sex, single year of age, cardiovascular state (neither, ischaemic heart disease, ischaemic stroke, both), diabetes status and BMI category (<25, 25 to <27, 27 to <30, ≥30). Hazards are calibrated to GBD 2023 and then change each year with GBD forecasts of IHD, IS and other-cause mortality, overweight and obesity, and diabetes, adjusted for the changing age composition of broad age groups with UN World Population Prospects 2024 (`code/preprocessing/build_trends.py`). Eligibility for GLP-1RA treatment (IHD or IS with BMI ≥27 or diabetes; diabetes at any BMI; BMI ≥30) is reassessed each year, and coverage is held constant over the 30 years. Dietary policy lowers first IHD and IS events, and for sodium also deaths after onset. All results, including scenarios, education groups and expenditure, are medians and 95% uncertainty intervals from the same 1000 Monte Carlo simulations.

## Running

```bash
bash code/pipeline/run_05_tables_figures.sh   # rebuild the supplementary tables and figures from the shipped results
bash code/pipeline/run_all.sh                 # full analysis from the raw data
python -m pytest code/lifetable -q            # unit tests of the life table
```

## Scripts

### code/

-  `code/lifetable_v3.py`  Main multi-state life-table entry point (baseline/glp1/diet/combo, IHD, stroke, T2D, CKD, allocation); writes the `--out` CSV plus `_byage`, `_t2d`, `_ckd` and `_check_*.csv`.

### code/pipeline/

-  `code/pipeline/env_main.sh`  Sets `LT_DP_OVERRIDE` and the single-thread BLAS variables; sourced by every pipeline and figure script.
-  `code/pipeline/run_01_dietary_inputs.sh`  Stage 1: builds the dose-response curves, dietary uncertainty and burdens, runs the dietary policy arm and the education PAFs; logs to `logs/s1_*.log`.
-  `code/pipeline/run_02_treatment_inputs.sh`  Stage 2: builds overweight prevalence, BMI selection/eligibility, CVD overweight, BMI27 parameters, income quintiles and affordability; logs to `logs/s2_*.log`.
-  `code/pipeline/run_03_monte_carlo.sh`  Stage 3: unit tests, main Monte Carlo (1000 draws x 27 coverage levels), curve quantiles, overlap, trajectories, education groups, T2D/CKD, scenario Monte Carlo (134 scenarios x 1000 draws), expenditure, the single-factor, secondary-outcome and discounted summaries (`scen_extra.py`) and the convergence check; progress lines start with `###`.
-  `code/pipeline/run_04_deterministic.sh`  Stage 4: deterministic runs used for model checks (coverage grid, decomposition, SELECT-like validation); published results come from stage 3; logs to `logs/s4_*.log`.
-  `code/pipeline/run_05_tables_figures.sh`  Stage 5: pre-aggregates the Monte Carlo output, builds the supplementary tables and renders Figures 1-6 and the supplementary figures; logs to `logs/s5_*.log`.
-  `code/pipeline/run_all.sh`  Master driver that runs stages 01-05, the trend conversion and pytest, with a parallel-jobs argument; writes logs under `logs/`.
-  `code/pipeline/verify_compare.py`  Compares pairs of CSV or XLSX files for numerical and textual equality; prints OK/DIFF per pair and exits non-zero on differences.
-  `code/pipeline/verify_release.sh`  End-to-end reproduction check: copies the release, links `data/`, reruns stages 1, 2, 4 and 5 and compares the outputs with `code/pipeline/verify_compare.py`.

### code/preprocessing/

-  `code/preprocessing/build_trends.py`  Converts the GBD forecasts in `data/input/forecasts/` (IHD and IS mortality, all-cause mortality, overweight and obesity, diabetes) into yearly factors relative to 2023, adjusting the broad age groups for their changing age composition with UN WPP 2024; writes `data/params/trends/cvd_factor.csv`, `oth_factor.csv`, `bmi_target.csv` and `dm_factor.csv`.
-  `code/preprocessing/add_nzl_quintiles.py`  Adds New Zealand income quintiles from the World Bank GNI file plus fixed survey shares; appends rows to `data/params/income_quintiles_wb.csv`.
-  `code/preprocessing/affordability_model.py`  Computes branded and generic GLP-1RA annual cost as a percentage of quintile income; writes `data/results/affordability_matrix.csv`.
   Branded prices in `data/params/glp1_prices.csv` are self-pay prices for one pen of semaglutide 2.4 mg (four weekly doses), accessed on 27 October 2026, with 13 pens a year, converted at ECB reference rates of 26 October 2026; the file records the local price, VAT, price type, exchange rate and source for each country.
-  `code/preprocessing/build_bmi27_params.py`  Derives adult BMI>=27/30 prevalence and CVD-specific overweight shares from GBD data; reads `data/params/select_elig.csv` and `data/gbd/cause_totals/`; writes `data/params/overweight_prevalence_bmi27.csv` and `data/params/elig_scale_bmi30_rel27.csv`.
-  `code/preprocessing/build_bmi_gamma.py`  Fits a gamma BMI distribution to p25/p30 to estimate BMI>=27 prevalence; reads `data/params/select_elig.csv` and `data/gbd/1-3_population.csv`; writes `data/params/overweight_prevalence_bmi27_gamma.csv` and `data/params/bmi27_fit_comparison.csv`.
-  `code/preprocessing/build_coverage_afford_cap.py`  Computes the maximum affordable GLP-1RA coverage per country from `data/results/affordability_matrix.csv` and `data/params/income_quintiles_global.csv`; writes `data/params/coverage_afford_cap.csv` (or compares with `--check`).
-  `code/preprocessing/build_curves.py`  Converts the GBD dose-response curve files into TMREL-anchored relative risks with the gdd_var mapping; writes `data/params/dose_response_conv_fixed.csv`.
-  `code/preprocessing/build_diet_uncertainty.py`  Extracts GBD 2023 diet-attributable death percentages with uncertainty intervals from `data/gbd/**`; writes `data/params/diet_paf_uncertainty.csv`.
-  `code/preprocessing/build_factor_burden.py`  Extracts single-factor diet-attributable PAFs and deaths by country, sex, age and outcome from `data/gbd/`; writes `data/results/factor_burden_by_age.csv`.
-  `code/preprocessing/build_global_income_quintiles.py`  Builds global income quintiles from World Bank GNI/share files and WPP population; writes `data/params/income_quintiles_global.csv`.
-  `code/preprocessing/build_joint_burden.py`  Computes joint diet-attributable deaths and crude/age-standardised rates for six outcomes; writes `data/results/joint_burden_2023.csv` and `data/results/supp_table2_joint.csv`.
-  `code/preprocessing/build_overweight_cvd.py`  Estimates overweight prevalence among IHD/stroke cases using GBD BMI exposure and cohort RRs; writes `data/params/overweight_cvd.csv` and `data/params/overweight_cvd_sensitivity.csv`.
-  `code/preprocessing/build_overweight_prevalence.py`  Computes adult BMI>=25 prevalence per country from GBD BMI exposure and population files; writes `data/params/overweight_prevalence.csv`.
-  `code/preprocessing/build_select_elig.py`  Fits a log-normal BMI distribution per sex/age from GBD prevalence and T2D rates; writes `data/params/select_elig.csv`.
-  `code/preprocessing/diet_arm_parallel.py`  Parallel driver computing the dietary PAFs for all policy scenarios and single factors; writes `data/results/v3/diet_policy_arm_v2_tailored.csv`.
-  `code/preprocessing/diet_policy_arm.py`  Computes baseline and scenario dietary PAFs (feasible, aspirational, tailored, single-factor) with the sodium-via-SBP effects; intakes move towards the TMREL only from the unfavourable side, so protective foods are never lowered and harmful factors never raised; also defines the parts of the feasible package used for the effect on deaths after onset (`post_na`, `post_na_wg`).
-  `code/preprocessing/equity_paf_by_edu_v2.py`  Computes the dietary PAFs by education tier using cross-classified exposures; writes `data/results/v3/equity_paf_by_edu_v2.csv`.
-  `code/preprocessing/equity_paf_by_edu_mc.py`  Low-to-high education PAF ratios with 95% uncertainty intervals from 1000 draws of the GDD education gradient, by the same method as `equity_paf_by_edu_v2.py` (the central estimate equals the ratio of its PAFs); writes `data/results/v3/equity_paf_ratio_ui.csv` (Supplementary Table 13, Figure 6a).
-  `code/preprocessing/paf_deficiency_v2.py`  Computes the age-specific dietary PAFs from the dose-response curves and exposure with a log-normal CV distribution and channel convergence; writes `data/results/paf_denominator.csv`.
-  `code/preprocessing/parse_income_quintiles.py`  Parses the World Bank income-share and GNI bulk files for the 19 countries; writes `data/params/income_quintiles_wb.csv`.
-  `code/preprocessing/select_check_v3.py`  Simulates a SELECT-like cohort with the life table to compare against trial mortality and event ratios; writes `data/results/v3/select_check.csv`.

### code/analysis/

-  `code/analysis/convergence_check.py`  Compares the first and second halves of the Monte Carlo draws for medians and UI bounds; writes `data/results/v3/mc/convergence_halves.csv`.
-  `code/analysis/cost_v3.py`  Computes GLP-1RA versus dietary-policy expenditure per death averted from the programme/allocation files; writes `tableS_cost_v3.csv` in `--outdir`.
-  `code/analysis/coverage_check.py`  Checks annual enrolment against the coverage targets in `data/results/v3/grid/**/*_check_programme.csv`; writes `coverage_check_by_year.csv` and `coverage_check_summary.csv`.
-  `code/analysis/equity_cost_v3.sh`  Runs `code/lifetable_v3.py` at 10%, 20% and affordable coverage, then `code/analysis/equity_v3.py` and `code/analysis/cost_v3.py`; writes `data/results/v3/equity_cost_final/cov*` tables.
-  `code/analysis/equity_v3.py`  Distributes deaths averted across education groups using Wittgenstein populations, education PAFs and affordability; writes `tableS_equity_v3.csv` and `tableS_equity_mapping_sensitivity_v3.csv`.
-  `code/analysis/equity_mc_aggregate.py`  Summarises the education allocation over the 1000 Monte Carlo draws (medians and 95% UIs, ratios within each draw); writes `data/results/v3/equity_mc/equity_ui.csv`, `pooled_ui.csv` and `mapping_ui.csv` (Supplementary Tables 17 and 18, Figure 6c, Supplementary Figure 9).
-  `code/analysis/expenditure_ui_v3.py`  Expenditure per death averted and the equal-budget comparison with 95% UIs from the 1000 main Monte Carlo draws, with deaths averted and person-years on treatment from each draw; writes `data/results/v3/expenditure_ui/cost_ui.csv` and `budget_ui.csv` (Supplementary Tables 19 to 21, Figure 6d, Supplementary Figure 10).
-  `code/analysis/grid_v3.sh`  Runs the life table over a coverage grid with risk-prioritised and uniform allocation; writes `data/results/v3/grid/*.csv` and `summary_crossover.csv`.
-  `code/analysis/mc_curve_quantiles.py`  Computes the median and 95% UI of the diet/treatment deaths-averted ratio by coverage from the Monte Carlo outputs; writes `data/results/v3/mc/curve_ratio_quantiles.csv`.
-  `code/analysis/mc_scen_v3.py`  Defines the 134 scenarios (including the trend, eligibility and dietary after-onset scenarios) and runs the life table per scenario, draw and coverage with the same 1000 draws as the main analysis; ratios and deaths are summarised with medians and linearly interpolated centiles, crossover with the nearest simulation values; writes `data/results/v3/mc_scen/summary_scenarios.csv` and `summary_paired.csv`.
-  `code/analysis/mc_scen_v3.sh`  Prepares the scenario job list, runs the `code/analysis/mc_scen_v3.py` workers in parallel and aggregates; writes `data/results/v3/mc_scen/summary_scenarios*.csv`.
-  `code/analysis/mc_traj_v3.py`  Runs the life table for the trajectory settings (10%, 20%, affordable, aspirational) per Monte Carlo draw; writes `trajectories.csv`, `trajectories_all_outcomes.csv` and `overtaking_year.csv`.
-  `code/analysis/mc_v3.py`  Main Monte Carlo: samples hazard ratios and persistence parameters, runs `code/lifetable_v3.py` per draw and coverage; writes `draws.csv`, `diet/draw*.csv`, `out/*.csv` and the summaries; with `MC_OUT` writes to another folder and with `MC_KEEP_BYAGE` also keeps the deaths averted by age (used by `run_equity_mc.sh`).
-  `code/analysis/mc_v3.sh`  Driver for the main Monte Carlo: prepares the draws, runs the `code/analysis/mc_v3.py` workers in parallel and aggregates `data/results/v3/mc/summary_*.csv`.
-  `code/analysis/outcome_crossover_v3.py`  Computes the crossover coverage and diet/treatment ratios for T2D and CKD from `data/results/v3/mc_t2dckd/out`; writes `outcome_curves.csv` and `summary_outcome_crossover.csv`.
-  `code/analysis/outcome_cumulative_v3.py`  Aggregates 30-year cumulative totals and effect-overlap gains by outcome from `data/results/v3/mc_traj_t2dckd/out`; writes `cumulative_by_outcome.csv` and `overlap_by_outcome.csv`.
-  `code/analysis/overlap_v3.py`  Quantifies the incremental benefit of combining diet and treatment at 0-100% effect overlap; writes `data/results/v3/overlap/overlap_by_country.csv` and `overlap_summary.csv`.
-  `code/analysis/rank_tailored_factors.py`  Ranks single-factor dietary benefits per country from `data/results/v3/single_factor/single_v*.csv`; writes `data/params/tailored_top3.csv` (or verifies with `--check`).
-  `code/analysis/run_decomp_v3.sh`  Runs the main, postA and postC life-table variants for the decomposition; writes `data/results/v3/decomp/decomp_summary.csv` and `main_check_programme.csv`.
-  `code/analysis/run_equity_mc.sh`  Reruns the 1000 Monte Carlo draws at 20% coverage keeping deaths averted by age (checking that they reproduce the main Monte Carlo exactly), applies `equity_v3.py` to each draw and summarises with `equity_mc_aggregate.py`; writes `data/results/v3/equity_mc/`.
-  `code/analysis/scen_extra.py`  From the per-draw outputs, the single-factor deaths and shares (Supplementary Table 10, Figure 4e), T2D and CKD in the secondary scenarios (Supplementary Table 22), discounted expenditure (Supplementary Table 19) and the combined effect multiples for T2D and CKD (Supplementary Table 8).
-  `code/analysis/secondary_ui_v3.py`  Earlier version of the secondary summaries, superseded by `scen_extra.py`.
-  `code/analysis/run_single_factor_v3.sh`  Deterministic single-factor runs, used only to choose the factors of the country-tailored packages (`rank_tailored_factors.py`); published single-factor results come from the scenario Monte Carlo.
-  `code/analysis/run_t2d_ckd.sh`  Re-runs the coverage grid and trajectories with T2D and CKD enabled; writes `data/results/v3/mc_t2dckd` and `data/results/v3/mc_traj_t2dckd` outputs.
-  `code/analysis/sa_t2d_ckd.sh`  Deterministic version of the T2D and CKD sensitivity analyses, superseded by the `sec_*` scenarios of the scenario Monte Carlo.
-  `code/analysis/test_select_v3.sh`  External validation of a SELECT-like cohort under several hazard variants; writes `data/results/v3/select_main.csv` and `select_*.csv`.

### code/outputs/

-  `code/outputs/build_supplementary_tables_v5.py`  Rebuilds Supplementary Tables 1 to 22 from `data/results/v3/*` into `data/tables/Supplementary_Tables_v5.xlsx` using the v4 template, restructures and renumbers the tables, and applies the published table notes (`data/tables/published_table_notes.json`), whose wording was edited for publication; all numbers and the table layout are generated.
-  `code/outputs/verify_published_tables.py`  Checks, cell by cell, that the generated workbook matches the published supplementary tables (`data/tables/Supplementary_Tables_published.xlsx`).
-  `code/outputs/make_single_factor_table.py`  Writes the single-factor shares from the scenario Monte Carlo in the format read by Figure 4e (`data/tables/table_single_factor_mc.csv`).
-  `code/outputs/check_burden_tables.py`  Recomputes diet-attributable cardiovascular deaths from `data/results/factor_burden_by_age.csv` and compares them with Supplementary Table 5 in the workbook.
-  `code/outputs/plot_figure1_starting_points.py`  Figure 1 three-panel starting-point plot (burden, eligibility, dietary factor shares); reads Supplementary Tables 2 and 6, the Monte Carlo crossover summary and the decomp programme file; writes `data/figures/figure1_starting_points.pdf/png` and `_data.csv`.
-  `code/outputs/plot_figure2_projections.py`  Figure 2 annual and cumulative projections with country small multiples; reads the Monte Carlo trajectories, `summary_deaths`, overlap and scenario CSVs; writes `data/figures/figure2_projections*`.
-  `code/outputs/plot_figure3_threshold.py`  Figure 3 threshold curves, coverage heatmap and crossover panel; reads the Monte Carlo and scenario summaries, overlap and curve quantiles; writes `data/figures/figure3_threshold.pdf/png/_data.csv`.
-  `code/outputs/plot_figure4_country_drivers.py`  Figure 4 country-driver correlations and dietary priority bubble matrix; reads the supplementary tables, `table_single_factor_mc.csv` and the Monte Carlo summary; writes `data/figures/figure4_country_drivers*` and `_correlations.csv`.
-  `code/outputs/plot_figure5_robustness.py`  Figure 5 robustness forest plot and country heatmap across the sensitivity scenarios; reads the `data/results/v3/mc` and `mc_scen` summaries; writes `data/figures/figure5_robustness.pdf/png/_data.csv`.
-  `code/outputs/plot_figure6_equity.py`  Figure 6 (education gradient, affordability, cost threshold, deaths averted by education group) and Supplementary Figure 9; reads Supplementary Tables 13 and 16, `data/results/v3/expenditure_ui/budget_ui.csv` and `data/results/v3/equity_mc/`; writes `data/figures/figure6_equity*`.
-  `code/outputs/supp_preaggregate.py`  Pre-aggregates the Monte Carlo outputs into `data/results/v3/supp/prob_diet_leads.csv` and `data/results/v3/supp/outcomes_by_arm.csv`.
-  `code/outputs/plot_suppfig_expenditure.R`  Supplementary Figure 10 polar charts of expenditure per death averted with 95% UIs in ggplot2; reads `data/results/v3/expenditure_ui/cost_ui.csv`; writes `data/figures/figure6c_cost_log.*`.
-  `code/outputs/plot_suppfig_t2d_ckd.py`  Supplementary Figure 4 T2D and CKD trajectories plus crossover panels (annual T2D shown to 2050, the end of the GBD diabetes forecasts); reads the `mc_traj_t2dckd` and `mc_t2dckd` summaries; writes `data/figures/suppfig_t2d_ckd.pdf/png`.

### code/lifetable/

-  `code/lifetable/__init__.py`  Package marker with an overview of the life-table modules.
-  `code/lifetable/bmi.py`  BMI categories: initial distributions, eligibility pools by tier, yearly BMI targets from the forecasts and the yearly rebalancing of enrolled and non-enrolled adults together.
-  `code/lifetable/test_bmi.py`  Unit tests of the BMI structure (30 checks, including that rebalancing preserves selective enrolment and that T2D cases averted are never negative); `--model` also runs the full-model checks.
-  `code/lifetable/allocation.py`  Allocates treatment places to eligible adults by risk priority, uniform shares or tiers; returns the eligible pool, target and achieved numbers.
-  `code/lifetable/calibration.py`  Calibrates the four-state life-table and CKD rates to the GBD incidence and death targets; emits feasibility and calibration check rows.
-  `code/lifetable/ckd.py`  Runs the CKD extension with SBP-mediated effects for each arm; returns the rows of CKD deaths averted by year.
-  `code/lifetable/diabetes.py`  Builds the type 2 diabetes overlay (incidence, overweight/diabetes splits, tier-3 BMI-30 rule) and the diabetes hazard multipliers.
-  `code/lifetable/effects.py`  Builds the hazard multipliers for the dietary policy and GLP-1RA effects by arm, eligibility tier and ramp year; dietary sodium also lowers deaths after IHD and IS onset (`SENS_DIET_POST`, main analysis `post_na`), combined multiplicatively with treatment; `code/lifetable_v3.py` uses the sodium component of the dietary package being run (none for single factors other than sodium, and in the tailored package only where it includes sodium).
-  `code/lifetable/test_diet_post.py`  Tests the dietary effect on deaths after onset: no change without `SENS_DIET_POST`, and multiplicative combination with treatment.
-  `code/lifetable/test_diet_shift.py`  Tests that intakes move towards the TMREL only from the unfavourable side and that the feasible package is unchanged by this rule.
-  `code/lifetable/eligibility.py`  Computes the overweight shares by disease state, age-specific eligibility, coverage targets, initial states and open-population entrants.
-  `code/lifetable/inputs.py`  Loads and pickles the cached GBD rates and population, and reads the dietary PIF file for the selected scenario.
-  `code/lifetable/outputs.py`  Writes the result CSV plus the `_byage`, `_t2d`, `_ckd` and `_check_*.csv` diagnostics and prints the calibration/conservation summaries.
-  `code/lifetable/persistence.py`  Models treatment stopping and restarting with residual benefit, and reconstructs the published discontinuation/restart figures.
-  `code/lifetable/projection.py`  Runs the 30-year multi-arm projection per country (allocation, ageing, BMI rebalancing, events, forecast trends, diabetes calibration to the forecast prevalence, T2D overlay with survival as without intervention, persistence, entrants).
-  `code/lifetable/settings.py`  Defines the ages, sexes, states, arms and treatment states, and reads all `SENS_*` environment settings into a namespace.
-  `code/lifetable/test_lifetable.py`  pytest unit tests for the transitions, calibration, persistence, allocation and eligibility helpers in `code/lifetable`.
-  `code/lifetable/transitions.py`  Provides the hazard construction, scalar and vectorised one-year transition steps and age-one-year aggregation.

## Data

`data/` holds every input and output, with at most one subfolder level (the Monte Carlo tree under
`data/results/v3/` keeps its own level):

-  `data/input/`  processed GBD dietary exposure, systolic blood pressure and Wittgenstein education inputs, and in `data/input/forecasts/` the GBD forecast and UN WPP extracts used for the trends;
-  `data/params/`  model parameters: country map, dose-response curves, mediation matrix, trial effects, self-pay prices, income quintiles, overweight prevalence, exposures and the yearly trend factors (`data/params/trends/`);
-  `data/results/`  result summaries read by the figure and table scripts;
-  `data/tables/`  supplementary-table workbooks, including the published supplementary tables (`Supplementary_Tables_published.xlsx`), the published table notes and the single-factor table;
-  `data/README.md`  sources and download instructions for the raw data, which are not redistributed.

## Licence and citation

The code is released under the MIT licence (see `LICENSE`). Inputs derived from the Global Burden of Disease Study 2023, the Global Dietary Database, UN World Population Prospects 2024, the World Bank and the Wittgenstein Centre remain subject to the terms of those sources (see `data/README.md`). If you use this code, please cite the accompanying article in The BMJ.
