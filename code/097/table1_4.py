import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
import os
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Load residualized data
filepath = os.path.join(INPUT_DATA_DIR, '097/residuals_bycell_final.dta')
df = pd.read_stata(filepath)


# Clean data
required_vars = ['r_ln_gdelt_pc3', 'r_pct_cov3', 'r_covgdp_wb3', 'r_flash2_tv3', 'r_flash2_tvgdp3', 'pop_ip']
df = df.dropna(subset=required_vars)

# Convert logged y back to levels since replicate function logs internally
df['r_gdelt_pc3_level'] = np.exp(df['r_ln_gdelt_pc3'])

# Table 1 Column 4: IV regression on residualized variables
# Stata: ivreg2 r_ln_gdelt_pc3 (r_pct_cov3 r_covgdp_wb3 = r_flash2_tv3 r_flash2_tvgdp3) [aw=pop_ip], cluster(country)

# Define variables
y = df['r_gdelt_pc3_level']  # Dependent variable (exp of logged residuals)
X = df[['r_pct_cov3', 'r_covgdp_wb3']]  # Endogenous regressors
z = df[['r_flash2_tv3', 'r_flash2_tvgdp3']]  # Instruments
weights = df['pop_ip']  # Population weights

metadata = {
    'paper_id': '097',
    'table_id': '1', 
    'panel_identifier': '4',
    'model_type': 'IV-residualized',
    'comments': 'Table 1 Column 4: 2SLS on residualized variables (exp of logged y, re-logged internally)'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='r_pct_cov3',
    endog_x=['r_pct_cov3', 'r_covgdp_wb3'],
    z=z,
    #weights=weights,
    elasticity=False,
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)