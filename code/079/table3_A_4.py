import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '079/Estimation_data_set_short_run_wages.dta')
df = pd.read_stata(filepath)

# Drop rows with missing values in key variables
required_cols = ['Dwage_low_skilled', 'Dshare_mexicans_low_skilled', 
                 'Dln_exports', 'Dln_state_gdp', 
                 'Dln_labor_high_skilled', 'Dln_labor_low_skilled', 'obs_wages_low_skilled']
df = df.dropna(subset=required_cols)

# =====================================
# Table 3 Panel A Column 4: D Wage, OLS with controls
# =====================================

# Dependent variable - convert log difference to ratio
wage_ratio = np.exp(df['Dwage_low_skilled'])
y_col4 = wage_ratio

# X matrix: treatment + controls + constant
X_col4 = sm.add_constant(df[['Dshare_mexicans_low_skilled', 'Dln_exports', 'Dln_state_gdp', 
                              'Dln_labor_high_skilled', 'Dln_labor_low_skilled']], prepend=False)

# Weights
weights_col4 = df['obs_wages_low_skilled']

# METADATA FOR THIS RESULT
metadata_col4 = {
    'paper_id': '079',
    'table_id': '3',
    'panel_identifier': 'A_4',
    'model_type': 'log-linear',
    'comments': 'Table 3 Panel A Column 4: D Wage OLS with controls'
}

# RUN THE REPLICATION - OLS only, no IV
replicate(
    metadata=metadata_col4,
    y=y_col4,
    X=X_col4,
    interest='Dshare_mexicans_low_skilled',
    elasticity=False,
    fe=None,
    weights=weights_col4,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)