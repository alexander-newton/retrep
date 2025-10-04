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
# Table 2 Column 1: OLS labor payments, low education
# drop if edcat == 3 | edcat == 2 (keep only edcat == 1)
# areg d_lp dfb_data dfb_nt_data i.occ [aw = wt], absorb(czone_nt) vce(cluster statefip)
# =====================================

# Drop edcat == 3 | edcat == 2 (exactly as in Stata)
df_low = df[(df['edcat'] != 3) & (df['edcat'] != 2)].copy()

# Drop missing values for required variables
required_vars = ['d_lp', 'dfb_data', 'dfb_nt_data', 'occ', 'czone_nt', 'pop_czocc', 'statefip']
df_clean = df_low.dropna(subset=required_vars)

# IMPORTANT: d_lp is already in log form (log difference/change)
# The replicate function expects LEVEL data because it will log it internally
# Since d_lp = log(Y_t/Y_t-1), we need exp(d_lp) = Y_t/Y_t-1 (the level ratio)

# Dependent variable: Convert from log to level for the replicate function
y = np.exp(df_clean['d_lp'])  # Convert log difference to level ratio

# Main regressors: immigration shock INSTRUMENTS (dfb_data, dfb_nt_data)
main_vars = ['dfb_data', 'dfb_nt_data']

# Convert categorical variables to numeric if needed
df_clean['occ_numeric'] = pd.Categorical(df_clean['occ']).codes
df_clean['czone_nt_numeric'] = pd.Categorical(df_clean['czone_nt']).codes

# X matrix: main variables + FE variables
X = df_clean[main_vars + ['occ_numeric', 'czone_nt_numeric']].copy()

# Add constant
X = sm.add_constant(X, prepend=False)

# Get the indices of FE columns in X
fe_indices = [X.columns.get_loc('occ_numeric'), X.columns.get_loc('czone_nt_numeric')]

# Weights (using pop_czocc as in Stata: gen wt = pop_czocc)
weights = df_clean['pop_czocc']

# Cluster variable
cluster = df_clean['statefip']

# METADATA
metadata = {
    'paper_id': '103',
    'table_id': '2',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 2 Column 1: OLS of labor payments change on immigration shock instruments (dfb_data, dfb_nt_data) with occupation and CZ-NT group FE, weighted by pop_czocc, clustered by state'
}

# RUN THE REPLICATION
# Note: Pass both variables as interest since both are main regressors
replicate(
    metadata=metadata,
    y=y,  # Pass LEVEL data - replicate will handle log transformation
    X=X,
    interest='dfb_data',  # Both dfb_data and dfb_nt_data are variables of interest
    fe=fe_indices,
    elasticity=False,
    weights=weights,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
