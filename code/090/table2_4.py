# Table 2 Column 4 — Anderson et al. (2020): State-level, log DV, OLS, pweights=births, cluster by state
import yaml
import os, glob, sys
import pandas as pd
import statsmodels.api as sm
import numpy as np

# make your replication helper available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ---------- LOAD DATA ----------
# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '090/State-Level Maternal Mortality Analysis - Anderson et al. (2020).dta')
df = pd.read_stata(filepath).dropna(subset=['childbirth_puerperal_rate4', 'midwife_license_law'])


# ---------- SPEC: Table 2, Column (4) ----------
# Dependent variable
y = df['childbirth_puerperal_rate4']

# Treatment
treat = 'midwife_license_law'

# Policy controls
policy_controls = [
    'diploma_mandatory','prelim_educ','applicant_exam_baker','four_yr_study_baker','inferior_schools',
    'osteopath_lic','osteopath_lic_missing','nurse_prac_lic','nurse_prac_lic_missing','child_hygiene',
    'suffrage','st_dollars1930'
]

# Demographic controls (interpolated shares + physicians)
demo_controls = [
    'dless18_imp','d18_65_imp','dwhite_imp','dblack_imp','durban_imp','dfemale_imp',
    'dforeign_born_imp','phys_per_cap_imp'
]

# Water/milk quality proxies
env_proxies = ['typhoid_rate','other_tb_rate']  # (other_tb_rate = nonpulmonary TB)

# Fixed effects (handled by replicate via fe=)
fe_vars = ['state_fips','year']

# Build X (treatment first for readability)
X = sm.add_constant(df[[treat] + policy_controls + demo_controls + env_proxies + fe_vars], prepend=False)

# Cluster and weights
clusters = df['state_fips']
pweights = df['births']  # Stata [pweight = births]

# Metadata for your tracker
metadata = {
    'paper_id': '090',
    'table_id': '2',
    'panel_identifier': '4',
    'model_type': 'log-linear',
    'comments': 'Replicates Table 2, column (4): demographics + policies + typhoid + nonpulmonary TB'
}

# ---------- RUN ----------
# If your replicate() supports weights & clustering internally, pass via kwargs_ols; otherwise adapt as needed.
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=treat,
    elasticity=False,                    # DV already in logs
    fe=fe_vars,                          # i.state_fips + i.year
    weights=pweights,                 # pweight = births
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': clusters}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
