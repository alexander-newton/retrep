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
# Table I Column 2b: 2SLS employment change
# ivreg2 d_demp (dfb_data dfb_nt_data = dfb_occ_ed1 dfb_occ_ed2 dfb_occ_ed3 dfb_occ_ed1_nt dfb_occ_ed2_nt dfb_occ_ed3_nt) //
#        i.occ i.czone_nt [aw = wt], cluster(statefip) first
# =====================================

# Filter for low education (edcat == 1)
df_low = df[df['edcat'] == 1].copy()

# Drop missing values for required variables
required_vars = ['d_demp', 'dfb_data', 'dfb_nt_data', 'dfb_occ_ed1', 'dfb_occ_ed2', 'dfb_occ_ed3', 'dfb_occ_ed1_nt', 'dfb_occ_ed2_nt', 'dfb_occ_ed3_nt', 'occ', 'czone_nt', 'dpop_czedocc', 'statefip']
df_clean = df_low.dropna(subset=required_vars)

# Dependent variable: employment change (raw, not exponentiated)
y = np.exp(df_clean['d_demp'])

# Endogenous regressors: immigration shocks
endog_x = ['dfb_data', 'dfb_nt_data']

# X matrix: include endogenous variables
X = df_clean[endog_x].copy()

# Add constant
X = sm.add_constant(X, prepend=False)

# Add ALL fixed effects variables as columns to X
X['occ'] = df_clean['occ']
X['czone_nt'] = df_clean['czone_nt']

# Get the indices of FE columns in X
# X now has columns: ['dfb_data', 'dfb_nt_data', 'const', 'occ', 'czone_nt']
fe_indices = [X.columns.get_loc('occ'), X.columns.get_loc('czone_nt')]

# Instruments matrix: all 6 occupation-education interaction instruments
z = df_clean[['dfb_occ_ed1', 'dfb_occ_ed2', 'dfb_occ_ed3', 'dfb_occ_ed1_nt', 'dfb_occ_ed2_nt', 'dfb_occ_ed3_nt']].copy()
z = sm.add_constant(z, prepend=False)

# Weights
weights = df_clean['dpop_czedocc']

# Cluster variable (for future use when clustering is implemented)
cluster = df_clean['statefip']

# METADATA
metadata = {
    'paper_id': '103',
    'table_id': '1',
    'panel_identifier': 'B_2',
    'model_type': 'log-linear',
    'comments': 'Table I Column 2b: 2SLS of employment change on immigration shocks (low education) with occupation and commuting zone FE, clustered by state'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='dfb_data',
    endog_x=endog_x,
    z=z,
    fe=fe_indices,  # Pass indices of FE columns in X, not the data itself
    elasticity=False,
    weights=weights,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)