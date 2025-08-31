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

# LOAD DATA - cross-age dataset
filepath = os.path.join(INPUT_DATA_DIR, '079/est_long_run_age.dta')  
df = pd.read_stata(filepath)

# Drop rows with missing values in key variables
required_cols = ['wage_low_skilled_2000_1990', 'share_mexpop_2000_1990', 
                 'pred_share_mexicans_2000_1990', 'obs_1990']  
df = df.dropna(subset=required_cols)

# =====================================
# Table 9 Panel A Column 6: Cross-Age IV for Low-Skilled Wages
# =====================================

# Dependent variable - calculate wage ratio from wage change
# wage_low_skilled_2000_1990 represents the change in low-skilled wages
y_col6 = np.exp(df['wage_low_skilled_2000_1990'])  

# X matrix: endogenous variable + constant
X_col6 = sm.add_constant(df[['share_mexpop_2000_1990']], prepend=False)

# Instrument - predicted share based on age distribution
z_col6 = df[['pred_share_mexicans_2000_1990']]

# Weights
weights_col6 = df['obs_1990']

# METADATA FOR THIS RESULT
metadata_col6 = {
    'paper_id': '079',
    'table_id': '9',
    'panel_identifier': 'A_6',
    'model_type': 'log-linear',
    'comments': 'Table 9 Panel A Column 6: Cross-Age IV for Native Low-Skilled Wage. Long-run 2000-1990, 46 age categories'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col6,
    y=y_col6,
    X=X_col6,
    interest='share_mexpop_2000_1990',
    endog_x='share_mexpop_2000_1990',
    z=z_col6,
    elasticity=False,
    fe=None,
    weights=weights_col6,
    #output=True, output_dir=OUTPUT_DIR, replicated=True

)