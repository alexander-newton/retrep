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
# Table I Column 2b: 2SLS native employment, low education
# ivreg2 d_emp_native_low (immig_shock immig_shock_nontrad = immig_iv immig_iv_nontrad) //
#        i.occ i.czgroup [aw=pop1980], cluster(state)
# =====================================

# Filter for low education (edcat == 1)
df_low = df[df['edcat'] == 1].copy()

# Drop missing values for required variables
required_vars = ['d_demp', 'dfb', 'dfb_nt', 'dfb_data', 'dfb_nt_data', 'occ', 'czone_nt', 'pop_czocc', 'statefip']
df_clean = df_low.dropna(subset=required_vars)

# Dependent variable: native employment change (low education)
y = np.exp(df_clean['d_demp'])

# Endogenous regressors: immigration shocks
endog_x = ['dfb', 'dfb_nt']

# Convert categorical occ to numeric codes for FE algorithm
df_clean = df_clean.copy()
df_clean['occ_numeric'] = df_clean['occ'].cat.codes

# X matrix: include endogenous variables
X = df_clean[endog_x].copy()

# Add constant
X = sm.add_constant(X, prepend=False)

# Add the fixed effects column to X
X['occ_numeric'] = df_clean['occ_numeric']

# Get the indices of FE columns in X
# X now has columns: ['dfb', 'dfb_nt', 'const', 'occ_numeric']
# occ_numeric is at index 3
fe_indices = [X.columns.get_loc('occ_numeric')]  # This will be [3]

# Instruments matrix: just instruments (no need to include FE here)
z = df_clean[['dfb_data', 'dfb_nt_data']].copy()
z = sm.add_constant(z, prepend=False)

# Weights
weights = df_clean['pop_czocc']

# Cluster variable (for future use when clustering is implemented)
cluster = df_clean['statefip']

# METADATA
metadata = {
    'paper_id': '103',
    'table_id': '1',
    'panel_identifier': 'B_2',
    'model_type': 'log-linear',
    'comments': 'Table I Column 2b: 2SLS of native employment change on immigration shocks (low education) with occupation FE, clustered by state'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='dfb',
    endog_x=endog_x,
    z=z,
    fe=fe_indices,  # Pass indices of FE columns in X, not the data itself
    elasticity=False,
    #weights=weights
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)