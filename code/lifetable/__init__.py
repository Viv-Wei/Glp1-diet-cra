"""Multi-state life table comparing population dietary policy with GLP-1RA treatment programmes (BMJ, 19 countries).

Modules, in the order a projection uses them:
  settings       constants and the SENS_* settings of one run
  inputs         GBD 2023 rates, population, dietary policy file and other inputs
  calibration    joint prevalence of IHD and IS, calibration of baseline hazards, CKD calibration
  eligibility    BMI eligibility by disease state, sex and age; coverage target; initial states; people turning 25
  effects        treatment hazard ratios and the risk multipliers of each intervention arm
  diabetes       T2D overlay (new cases by arm) and the diabetes split of first-event incidence
  persistence    five treatment states (ON1, ON2, OFF1, OFF2, OFF3) among enrolled people
  allocation     risk-prioritised or uniform allocation of programme places
  transitions    hazards by state, annual competing-risk step, ageing
  projection     30-year projection of one country under the four arms
  ckd            separate two-state CKD model
  outputs        result and check tables
The command-line entry point is code/lifetable_v3.py.
"""
