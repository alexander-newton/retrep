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

# Check if file exists
if not os.path.exists(filepath):
    print(f"File not found at: {filepath}")
    print(f"Please ensure the data file is in the correct location")
    raise FileNotFoundError(f"Data file not found: {filepath}")

df = pd.read_stata(filepath)

# Drop rows with missing values in key variables
required_cols = ['Dwage_low_skilled_9294', 'mexican_shock', 'share_mexican_low_skilled_1980',
                 'wage_low_skilled_9193', 'Dln_state_gdp', 'Dln_exports', 
                 'Dln_labor_high_skilled', 'Dln_labor_low_skilled', 'obs_wages_low_skilled']
df = df.dropna(subset=required_cols)

# =====================================
# Table 3 Panel A Column 7: D Wage, Individual Controls, IV
# =====================================

# Dependent variable - convert log difference to ratio
# Since Dwage = log(wage_t/wage_t-1), exp(Dwage) = wage_t/wage_t-1
y_col7 = np.exp(df['Dwage_low_skilled_9294'])

# Controls - keep baseline wage in logs as in original
controls = ['wage_low_skilled_9193', 'Dln_state_gdp', 'Dln_exports', 
            'Dln_labor_high_skilled', 'Dln_labor_low_skilled']

# X matrix: endogenous variable + controls
X_col7 = sm.add_constant(df[['mexican_shock'] + controls], prepend=False)

# Instrument
z_col7 = df[['share_mexican_low_skilled_1980']]

# Weights
weights_col7 = df['obs_wages_low_skilled']

# METADATA FOR THIS RESULT
metadata_col7 = {
    'paper_id': '079',
    'table_id': '3',
    'panel_identifier': 'A_7',
    'model_type': 'log-linear',
    'comments': 'Table 3 Panel A Column 7: D Wage Individual Controls, IV. Y is wage ratio (exp of log difference)'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col7,
    y=y_col7,
    X=X_col7,
    interest='mexican_shock',
    endog_x='mexican_shock',
    z=z_col7,
    elasticity=False,
    fe=None,
    weights=weights_col7,
    #output=True, output_dir=OUTPUT_DIR, replicated=True

)