# Data

Raw downloads are not redistributed. Place them under `data/` with the names below (the scripts read these paths).

| Path | Source | Content |
|---|---|---|
| `data/gbd/cause_totals/queryB_ihd.csv`, `queryB_istroke.csv`, `queryB_t2d.csv`, `queryB_ckd_3years.csv`, `queryD_All_causes_deaths.csv` | GBD 2023 Results Tool (IHME) | incidence, prevalence and deaths, rates per 100 000, 2023, five-year age groups from 25 years, both sexes separately, the 19 countries and Global |
| `data/gbd/1-3_population.csv` | GBD 2023 population estimates | population by location, sex and five-year age group, 2023 |
| `data/gbd/<cause>/<risk>/IHME-GBD_2023_DATA-*/*.csv` | GBD 2023 Results Tool | deaths attributable to each dietary risk and to "Dietary risks" (joint), number and percent, for ischemic heart disease, ischemic stroke and diabetes mellitus type 2 |
| `data/gbd/queryA_ICH_all_merged.csv`, `queryA_SAH_all_merged.csv`, `queryA_ckd_3years.csv` | GBD 2023 Results Tool | the same for intracerebral and subarachnoid haemorrhage and chronic kidney disease |
| `data/gbd/curves/*.csv` | GBD 2023 Burden of Proof risk curves | log relative risk by exposure for each risk-outcome pair used (file names in `code/preprocessing/build_curves.py`) |
| `data/gbd/bmi_exposure/` | GBD 2023 high BMI exposure estimates | prevalence of overweight and obesity and mean BMI by location, sex and age, 2023 |
| `data/wpp/WPP2024_PopulationByAge5GroupSex_Medium.csv` | UN World Population Prospects 2024 | population by country, sex and five-year age group, medium variant |
| `data/worldbank/extracted/API_*.csv` | World Bank World Development Indicators (bulk download) | NY.GNP.PCAP.CD (GNI per capita, Atlas method, current US$) and SI.DST.FRST.20, SI.DST.02ND.20, SI.DST.03RD.20, SI.DST.04TH.20, SI.DST.05TH.20 (income shares by quintile), with their Metadata_Country files |
| `data/exposure_19country.csv` | Global Dietary Database (GDD 2018) | dietary intake by country, age, sex and education |

Supplied processed inputs: `data/input/exposure_gbd_{male,female}.csv` and `data/input/sbp_gbd2023_{male,female}.csv` (GBD 2023
dietary exposure and systolic blood pressure by country and age, as used by the dietary arm and the CKD model), and
`data/input/wcde_clean.csv` (education distribution by country, age and sex, Wittgenstein Centre).

New Zealand has no World Bank income shares; `code/preprocessing/add_nzl_quintiles.py` uses shares from the OECD Income Distribution
Database. GBD 2023 retained the dietary TMRELs of GBD 2021 (Table 8 of the GBD 2021 risk factor appendix), the values in
`data/params/tmrel_certified.csv`.
