import yaml
import os
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '103/workfile_immigration.dta')
df = pd.read_stata(filepath)

# =====================================
# Table I Column 1b: OLS with occupation and CZ-group fixed effects, weighted, clustered by state
# reg d_demp dfb dfb_nt i.occ i.czone_nt [aw=pop_czocc], cluster(statefip)
# =====================================

# Filter for low education (edcat == 1)
df_low = df[df['edcat'] == 1].copy()

# Drop missing values for required variables
required_vars = ['d_demp', 'dfb_data', 'dfb_nt_data', 'occ', 'czone_nt', 'dpop_czedocc', 'statefip']
df_clean = df_low.dropna(subset=required_vars)

# Dependent variable: native employment change (low education)
y = np.exp(df_clean['d_demp'])

# Convert categorical occ to numeric codes for FE algorithm (czone_nt is already numeric)
df_clean = df_clean.copy()
df_clean['occ_numeric'] = df_clean['occ'].cat.codes

# Main regressors: immigration shock INSTRUMENTS (dfb_data, dfb_nt_data)
main_vars = ['dfb_data', 'dfb_nt_data']

# X matrix: main variables + FE variables
X = df_clean[main_vars + ['occ_numeric', 'czone_nt']].copy()

# Add constant
X = sm.add_constant(X, prepend=False)

# Get the indices of FE columns in X
fe_indices = [X.columns.get_loc('occ_numeric'), X.columns.get_loc('czone_nt')]

# Weights (using dpop_czedocc as in Stata: gen wt = dpop_czedocc)
weights = df_clean['dpop_czedocc']

# Cluster variable (for future use when clustering is implemented)
cluster = df_clean['statefip']

# METADATA
metadata = {
    'paper_id': '103',
    'table_id': '1',
    'panel_identifier': 'B_1',
    'model_type': 'log-linear',
    'comments': 'Table I Column 1b: OLS of native employment change on immigration shock instruments (dfb_data, dfb_nt_data) with occupation and CZ-NT group FE, weighted by dpop_czedocc, clustered by state'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='dfb_data',
    fe=fe_indices,
    elasticity=False,
    weights=weights,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
