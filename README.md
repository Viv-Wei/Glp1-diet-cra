# Code and data

***GLP-1 receptor agonist treatment versus population dietary policy: 19-country comparative risk assessment***



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
-  `code/pipeline/run_03_monte_carlo.sh`  Stage 3: prepares and runs the Monte Carlo, curve quantiles, overlap, trajectories, T2D/CKD, scenario Monte Carlo and the convergence checks; logs to `logs/s3_*.log`.
-  `code/pipeline/run_04_deterministic.sh`  Stage 4: single-factor runs, coverage grid and checks, decomposition, SELECT validation, T2D/CKD sensitivity, equity and cost; logs to `logs/s4_*.log`.
-  `code/pipeline/run_05_tables_figures.sh`  Stage 5: pre-aggregates the Monte Carlo output, builds the supplementary tables and renders Figures 1-6 and the supplementary figures; logs to `logs/s5_*.log`.
-  `code/pipeline/run_all.sh`  Master driver that runs stages 01-05 plus pytest, with a parallel-jobs argument; writes logs under `logs/`.
-  `code/pipeline/verify_compare.py`  Compares pairs of CSV or XLSX files for numerical and textual equality; prints OK/DIFF per pair and exits non-zero on differences.
-  `code/pipeline/verify_release.sh`  End-to-end reproduction check: copies the release, links `data/`, reruns stages 1, 2, 4 and 5 and compares the outputs with `code/pipeline/verify_compare.py`.

### code/preprocessing/

-  `code/preprocessing/add_nzl_quintiles.py`  Adds New Zealand income quintiles from the World Bank GNI file plus fixed survey shares; appends rows to `data/params/income_quintiles_wb.csv`.
-  `code/preprocessing/affordability_model.py`  Computes branded and generic GLP-1RA annual cost as a percentage of quintile income; writes `data/results/affordability_matrix.csv`.
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
-  `code/preprocessing/diet_policy_arm.py`  Computes baseline and scenario dietary PAFs (feasible, aspirational, tailored, single-factor) with the sodium-via-SBP effects; writes `data/results/diet_policy_arm.csv`.
-  `code/preprocessing/equity_paf_by_edu_v2.py`  Computes the dietary PAFs by education tier using cross-classified exposures; writes `data/results/v3/equity_paf_by_edu_v2.csv`.
-  `code/preprocessing/paf_deficiency_v2.py`  Computes the age-specific dietary PAFs from the dose-response curves and exposure with a log-normal CV distribution and channel convergence; writes `data/results/paf_denominator.csv`.
-  `code/preprocessing/parse_income_quintiles.py`  Parses the World Bank income-share and GNI bulk files for the 19 countries; writes `data/params/income_quintiles_wb.csv`.
-  `code/preprocessing/select_check_v3.py`  Simulates a SELECT-like cohort with the life table to compare against trial mortality and event ratios; writes `data/results/v3/select_check.csv`.

### code/analysis/

-  `code/analysis/convergence_check.py`  Compares the first and second halves of the Monte Carlo draws for medians and UI bounds; writes `data/results/v3/mc/convergence_halves.csv`.
-  `code/analysis/cost_v3.py`  Computes GLP-1RA versus dietary-policy expenditure per death averted from the programme/allocation files; writes `tableS_cost_v3.csv` in `--outdir`.
-  `code/analysis/coverage_check.py`  Checks annual enrolment against the coverage targets in `data/results/v3/grid/**/*_check_programme.csv`; writes `coverage_check_by_year.csv` and `coverage_check_summary.csv`.
-  `code/analysis/equity_cost_v3.sh`  Runs `code/lifetable_v3.py` at 10%, 20% and affordable coverage, then `code/analysis/equity_v3.py` and `code/analysis/cost_v3.py`; writes `data/results/v3/equity_cost_final/cov*` tables.
-  `code/analysis/equity_v3.py`  Distributes deaths averted across education groups using Wittgenstein populations, education PAFs and affordability; writes `tableS_equity_v3.csv` and `tableS_equity_mapping_sensitivity_v3.csv`.
-  `code/analysis/grid_v3.sh`  Runs the life table over a coverage grid with risk-prioritised and uniform allocation; writes `data/results/v3/grid/*.csv` and `summary_crossover.csv`.
-  `code/analysis/mc_curve_quantiles.py`  Computes the median and 95% UI of the diet/treatment deaths-averted ratio by coverage from the Monte Carlo outputs; writes `data/results/v3/mc/curve_ratio_quantiles.csv`.
-  `code/analysis/mc_scen_v3.py`  Defines the sensitivity scenarios and runs the life table per scenario, draw and coverage; writes `data/results/v3/mc_scen/summary_scenarios.csv` and `summary_paired.csv`.
-  `code/analysis/mc_scen_v3.sh`  Prepares the scenario job list, runs the `code/analysis/mc_scen_v3.py` workers in parallel and aggregates; writes `data/results/v3/mc_scen/summary_scenarios*.csv`.
-  `code/analysis/mc_traj_v3.py`  Runs the life table for the trajectory settings (10%, 20%, affordable, aspirational) per Monte Carlo draw; writes `trajectories.csv`, `trajectories_all_outcomes.csv` and `overtaking_year.csv`.
-  `code/analysis/mc_v3.py`  Main Monte Carlo: samples hazard ratios and persistence parameters, runs `code/lifetable_v3.py` per draw and coverage; writes `draws.csv`, `diet/draw*.csv`, `out/*.csv` and `summary_crossover_ratios.csv`.
-  `code/analysis/mc_v3.sh`  Driver for the main Monte Carlo: prepares the draws, runs the `code/analysis/mc_v3.py` workers in parallel and aggregates `data/results/v3/mc/summary_*.csv`.
-  `code/analysis/outcome_crossover_v3.py`  Computes the crossover coverage and diet/treatment ratios for T2D and CKD from `data/results/v3/mc_t2dckd/out`; writes `outcome_curves.csv` and `summary_outcome_crossover.csv`.
-  `code/analysis/outcome_cumulative_v3.py`  Aggregates 30-year cumulative totals and effect-overlap gains by outcome from `data/results/v3/mc_traj_t2dckd/out`; writes `cumulative_by_outcome.csv` and `overlap_by_outcome.csv`.
-  `code/analysis/overlap_v3.py`  Quantifies the incremental benefit of combining diet and treatment at 0-100% effect overlap; writes `data/results/v3/overlap/overlap_by_country.csv` and `overlap_summary.csv`.
-  `code/analysis/rank_tailored_factors.py`  Ranks single-factor dietary benefits per country from `data/results/v3/single_factor/single_v*.csv`; writes `data/params/tailored_top3.csv` (or verifies with `--check`).
-  `code/analysis/run_decomp_v3.sh`  Runs the main, postA and postC life-table variants for the decomposition; writes `data/results/v3/decomp/decomp_summary.csv` and `main_check_programme.csv`.
-  `code/analysis/run_single_factor_v3.sh`  Runs the 12 single-factor dietary scenarios at 20% coverage in parallel; writes `data/results/v3/single_factor/single_v*.csv` and `data/tables/table_single_factor_v3.csv`.
-  `code/analysis/run_t2d_ckd.sh`  Re-runs the coverage grid and trajectories with T2D and CKD enabled; writes `data/results/v3/mc_t2dckd` and `data/results/v3/mc_traj_t2dckd` outputs.
-  `code/analysis/sa_t2d_ckd.sh`  Sensitivity analyses varying the T2D incidence hazard ratio and the blood-pressure fall at 20% and affordable coverage; writes `data/results/v3/sa_t2d_ckd/summary_sa_t2d_ckd.csv`.
-  `code/analysis/test_select_v3.sh`  External validation of a SELECT-like cohort under several hazard variants; writes `data/results/v3/select_main.csv` and `select_*.csv`.

### code/outputs/

-  `code/outputs/build_supplementary_tables_v5.py`  Rebuilds all supplementary tables from `data/results/v3/*` and `data/tables/table_single_factor_v3.csv` into `data/tables/Supplementary_Tables_v5.xlsx` using the v4 template.
-  `code/outputs/check_burden_tables.py`  Recomputes diet-attributable cardiovascular deaths from `data/results/factor_burden_by_age.csv` and compares them with Supplementary Table 5 in the workbook.
-  `code/outputs/figure1/plot_figure1_starting_points.py`  Figure 1 three-panel starting-point plot (burden, eligibility, dietary factor shares); reads Supplementary Tables 2 and 6, the Monte Carlo crossover summary and the decomp programme file; writes `data/figures/figure1_starting_points.pdf/png` and `_data.csv`.
-  `code/outputs/figure2/plot_figure2_projections.py`  Figure 2 annual and cumulative projections with country small multiples; reads the Monte Carlo trajectories, `summary_deaths`, overlap and scenario CSVs; writes `data/figures/figure2_projections*`.
-  `code/outputs/figure3/plot_figure3_threshold.py`  Figure 3 threshold curves, coverage heatmap and crossover panel; reads the Monte Carlo and scenario summaries, overlap and curve quantiles; writes `data/figures/figure3_threshold.pdf/png/_data.csv`.
-  `code/outputs/figure4/plot_figure4_country_drivers.py`  Figure 4 country-driver correlations and dietary priority bubble matrix; reads the supplementary tables, `table_single_factor_v3.csv` and the Monte Carlo summary; writes `data/figures/figure4_country_drivers*` and `_correlations.csv`.
-  `code/outputs/figure5/plot_figure5_robustness.py`  Figure 5 robustness forest plot and country heatmap across the sensitivity scenarios; reads the `data/results/v3/mc` and `mc_scen` summaries; writes `data/figures/figure5_robustness.pdf/png/_data.csv`.
-  `code/outputs/figure6/plot_figure6_equity.py`  Figure 6 education-gradient, affordability and price-scenario slope panels; reads the supplementary tables and `data/results/v3/equity_cost_final/cov0.20`; writes `data/figures/figure6_equity*`.
-  `code/outputs/supp_preaggregate.py`  Pre-aggregates the Monte Carlo outputs into `data/results/v3/supp/prob_diet_leads.csv` and `data/results/v3/supp/outcomes_by_arm.csv`.
-  `code/outputs/supplementary/plot_suppfig_expenditure.R`  Supplementary Figure 10 polar cost-per-death charts in ggplot2; reads `data/results/v3/equity_cost_final/cov0.20/tableS_cost_v3.csv`; writes `data/figures/figure6c_cost_log.*` and `data/figures/`.
-  `code/outputs/supplementary/plot_suppfig_t2d_ckd.py`  Supplementary Figure 4 T2D and CKD trajectories plus crossover panels; reads the `mc_traj_t2dckd` and `mc_t2dckd` summaries; writes `data/figures/suppfig_t2d_ckd.pdf/png`.

### code/lifetable/

-  `code/lifetable/__init__.py`  Empty package marker for the `code/lifetable` module.
-  `code/lifetable/allocation.py`  Allocates treatment places to eligible adults by risk priority, uniform shares or tiers; returns the eligible pool, target and achieved numbers.
-  `code/lifetable/calibration.py`  Calibrates the four-state life-table and CKD rates to the GBD incidence and death targets; emits feasibility and calibration check rows.
-  `code/lifetable/ckd.py`  Runs the CKD extension with SBP-mediated effects for each arm; returns the rows of CKD deaths averted by year.
-  `code/lifetable/diabetes.py`  Builds the type 2 diabetes overlay (incidence, overweight/diabetes splits, tier-3 BMI-30 rule) and the diabetes hazard multipliers.
-  `code/lifetable/effects.py`  Builds the hazard multipliers for the dietary policy and GLP-1RA effects by arm, eligibility tier and ramp year.
-  `code/lifetable/eligibility.py`  Computes the overweight shares by disease state, age-specific eligibility, coverage targets, initial states and open-population entrants.
-  `code/lifetable/inputs.py`  Loads and pickles the cached GBD rates and population, and reads the dietary PIF file for the selected scenario.
-  `code/lifetable/outputs.py`  Writes the result CSV plus the `_byage`, `_t2d`, `_ckd` and `_check_*.csv` diagnostics and prints the calibration/conservation summaries.
-  `code/lifetable/persistence.py`  Models treatment stopping and restarting with residual benefit, and reconstructs the published discontinuation/restart figures.
-  `code/lifetable/projection.py`  Runs the 30-year multi-arm projection per country (allocation, ageing, events, T2D overlay, persistence, entrants).
-  `code/lifetable/settings.py`  Defines the ages, sexes, states, arms and treatment states, and reads all `SENS_*` environment settings into a namespace.
-  `code/lifetable/test_lifetable.py`  pytest unit tests for the transitions, calibration, persistence, allocation and eligibility helpers in `code/lifetable`.
-  `code/lifetable/transitions.py`  Provides the hazard construction, scalar and vectorised one-year transition steps and age-one-year aggregation.

## Data

`data/` holds every input and output, with at most one subfolder level (the Monte Carlo tree under
`data/results/v3/` keeps its own level):

-  `data/input/`  processed GBD dietary exposure, systolic blood pressure and Wittgenstein education inputs;
-  `data/params/`  model parameters: country map, dose-response curves, mediation matrix, trial effects, prices, income quintiles, overweight prevalence and exposures;
-  `data/results/`  result summaries read by the figure and table scripts;
-  `data/tables/`  supplementary-table workbooks and the single-factor table;
-  `data/README.md`  sources and download instructions for the raw data, which are not redistributed.
