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
filepath = os.path.join(INPUT_DATA_DIR, '062/clean_IFLS.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 4 Column 5: Determinants of Bride Price Payment Amounts
# Dependent Variable: Log Bride Price Amount
# Full specification with all controls
# =====================================

# DATA PREPARATION
df = df[df['eth_group'] != 9].copy()

# Create log(1+pre_marriage_asset) to handle zeros
df['alt_log_pre'] = np.log(1 + df['pre_marriage_asset'])

# Only filter on dowry_gdp missing - this is the key filter from Stata
df = df[df['dowry_gdp'].notna()].copy()

# Filter to observations with all required variables
keep_cols = ['dowry_value', 
             'elementary_plus', 'junior_plus', 'college',
             'mar_age', 'mar_age2',
             'husband_elementary', 'husband_junior', 'husband_college',
             'husband_mar_age', 'hus_mar2',
             'alt_log_pre',
             'muslim', 'multiple_wives',
             'eth_group', 'year', 'mar_year', 'mar_year2']
df = df[keep_cols].dropna()

df = df[df['dowry_value'] > 0].copy()
df = df.reset_index(drop=True)

# Dependent variable
y_col5 = df['dowry_value']

# Main treatment variables
main_treatments = ['elementary_plus', 'junior_plus', 'college']

# All controls
controls = ['mar_age', 'mar_age2', 'husband_elementary', 'husband_junior', 
            'husband_college', 'husband_mar_age', 'hus_mar2', 'alt_log_pre',
            'muslim', 'multiple_wives', 'mar_year', 'mar_year2']

# X matrix with controls and categorical FE variables (NOT dummies)
X_col5 = pd.concat([
    df[main_treatments],
    df[controls],
    df[['eth_group', 'year']]  # Keep as categorical
], axis=1)

X_col5 = sm.add_constant(X_col5, prepend=False)

# Get indices of FE columns
fe_indices = [X_col5.columns.get_loc('eth_group'), X_col5.columns.get_loc('year')]

# METADATA
metadata_col5 = {
    'paper_id': '062',
    'table_id': '4',
    'panel_identifier': '5',
    'model_type': 'log-linear',
    'comments': 'Table 4 Column 5: Full specification with all controls. Robust SE.'
}

print(f"Observations before FE transformation: {len(y_col5)}")

# RUN REPLICATION
replicate(
    metadata=metadata_col5,
    y=y_col5,
    X=X_col5,
    interest='elementary_plus',
    fe=fe_indices,
    elasticity=False,

)