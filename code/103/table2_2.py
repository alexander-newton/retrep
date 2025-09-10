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
# Table 2 Column 2: 2SLS labor payments, low education
# drop if edcat == 3 | edcat == 2 (keep only edcat == 1)
# ivreg2 d_lp (dfb_data dfb_nt_data = dfb_occ_ed1 dfb_occ_ed2 dfb_occ_ed3 dfb_occ_ed1_nt dfb_occ_ed2_nt dfb_occ_ed3_nt) ///
#        i.occ i.czone_nt [aw = wt], cluster(statefip) first
# =====================================

# Filter for low education (edcat == 1) - drop edcat == 2 and 3
df_low = df[df['edcat'] == 1].copy()

# Drop missing values for required variables
required_vars = ['d_lp', 'dfb_data', 'dfb_nt_data', 'dfb_occ_ed1', 'dfb_occ_ed2', 'dfb_occ_ed3', 
                 'dfb_occ_ed1_nt', 'dfb_occ_ed2_nt', 'dfb_occ_ed3_nt', 'occ', 'czone_nt', 'pop_czocc', 'statefip']
df_clean = df_low.dropna(subset=required_vars)

# Dependent variable: labor payments change (low education)
y = np.exp(df_clean['d_lp'])

# Endogenous regressors: immigration shock instruments
endog_x = ['dfb_data', 'dfb_nt_data']

# Convert categorical occ to numeric codes for FE algorithm (czone_nt is already numeric)
df_clean = df_clean.copy()
df_clean['occ_numeric'] = df_clean['occ'].cat.codes

# X matrix: include endogenous variables + FE variables
X = df_clean[endog_x + ['occ_numeric', 'czone_nt']].copy()

# Add constant
X = sm.add_constant(X, prepend=False)

# Get the indices of FE columns in X
fe_indices = [X.columns.get_loc('occ_numeric'), X.columns.get_loc('czone_nt')]

# Instruments matrix: all education-specific instruments
z_vars = ['dfb_occ_ed1', 'dfb_occ_ed2', 'dfb_occ_ed3', 'dfb_occ_ed1_nt', 'dfb_occ_ed2_nt', 'dfb_occ_ed3_nt']
z = df_clean[z_vars + ['occ_numeric', 'czone_nt']].copy()
z = sm.add_constant(z, prepend=False)

# Weights (using pop_czocc as in Stata: gen wt = pop_czocc)
weights = df_clean['pop_czocc']

# Cluster variable (for future use when clustering is implemented)
cluster = df_clean['statefip']

# METADATA
metadata = {
    'paper_id': '103',
    'table_id': '2',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comments': 'Table 2 Column 2: 2SLS of labor payments change on immigration shock instruments (dfb_data, dfb_nt_data) with occupation and CZ-NT group FE, weighted by pop_czocc, clustered by state'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='dfb_data',
    endog_x=endog_x,
    z=z,
    fe=fe_indices,
    elasticity=False,
    #weights=weights
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)
