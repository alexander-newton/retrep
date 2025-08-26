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

# LOAD DATA - Zambia ZFPS
filepath = os.path.join(INPUT_DATA_DIR, '062/ZFPS_clean.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 4 Column 11: Determinants of Bride Price Payment Amounts
# Dependent Variable: Log Bride Price Amount
# Sample: Zambia ZFPS, bride price charged observations
# =====================================

# DATA PREPARATION
df = df[df['H_lobolacharged'] == 1].copy()
df['property'] = (df['property'] == 'Yes').astype(int)
df['land'] = (df['land'] == 'Yes').astype(int)


# Keep only needed variables - all exist in dataset
keep_cols = ['H_lobolacharged_amnt',
            'primary', 'junior_sec', 'secondary',
            'marriage_age', 'marriage_age2',
            'H_primary', 'H_junior_sec', 'H_secondary',
            'H_marriage_age', 'H_marriage_age2',
            'property', 'land',
            'ethnicity', 'mar_year', 'mar_year2']

df = df[keep_cols].dropna()
df = df[df['H_lobolacharged_amnt'] > 0].copy()
df = df.reset_index(drop=True)

# Dependent variable
y_col11 = df['H_lobolacharged_amnt']

# Main treatment variables
main_treatments = ['primary', 'junior_sec', 'secondary']

# Controls
controls = ['marriage_age', 'marriage_age2',
           'H_primary', 'H_junior_sec', 'H_secondary',
           'H_marriage_age', 'H_marriage_age2',
           'property', 'land',
           'mar_year', 'mar_year2']

# X matrix with controls and ethnicity as categorical (NOT dummies)
X_col11 = pd.concat([
   df[main_treatments],
   df[controls],
   df[['ethnicity']]  # Keep as categorical
], axis=1)

X_col11 = sm.add_constant(X_col11, prepend=False)

# Get index of ethnicity column for FE
fe_indices = [X_col11.columns.get_loc('ethnicity')]

# METADATA
metadata_col11 = {
   'paper_id': '062',
   'table_id': '4',
   'panel_identifier': '11',
   'model_type': 'log-linear',
   'comments': 'Table 4 Column 11: ZFPS Zambia hedonic regression. Full specification with all controls. Robust SE.'
}

print(f"Observations before FE transformation: {len(y_col11)}")

# RUN REPLICATION
replicate(
   metadata=metadata_col11,
   y=y_col11,
   X=X_col11,
   interest='primary',
   fe=fe_indices,
   elasticity=False,
   output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True

)