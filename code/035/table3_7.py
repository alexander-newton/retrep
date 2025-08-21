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
filepath = os.path.join(INPUT_DATA_DIR, '035/MozNatRes_Mozambique.dta')
df = pd.read_stata(filepath)

# =====================================
# CREATE MISSING VARIABLES BEFORE FILTERING
# =====================================

# Convert ONLY specific categorical variables that need to be numeric
# NOT year! Keep year as is for filtering
vars_to_convert = ['tr_0', 'tr_1', 'tr_2', 'tr_3', 'ethn_macua', 'ethn_maconde', 
                   'educ_1', 'educ_2', 'educ_3', 'rel_muslim', 'married', 
                   'sub_farmer', 'a16'] + \
                  [f'cm_res_{x}' for x in ['limestone', 'marble', 'sand', 'forest', 
                   'ebony', 'exwood', 'gold', 'charcoal', 'graphite', 'stones', 
                   'mercury', 'fishing', 'salt', 'nat_gas']]

for col in vars_to_convert:
    if col in df.columns and df[col].dtype.name == 'category':
        df[col] = df[col].cat.codes

# 1. Try to create infrastructure if variables exist
infra_vars = ['cm_a1_a_1', 'cm_a1_b_1', 'cm_a1_c_1', 'cm_a1_d_1', 
              'cm_a1_e_3', 'cm_a1_f_3', 'cm_a1_g_3', 'cm_a1_h_3', 
              'cm_a1_i_3', 'cm_a1_j_3', 'cm_a1_k_3', 'cm_a1_l_3', 
              'cm_a1_m', 'cm_a1_n']

existing_infra = [var for var in infra_vars if var in df.columns]
if existing_infra:
    # Convert infrastructure variables to numeric first
    for var in existing_infra:
        if df[var].dtype.name == 'category':
            df[var] = df[var].cat.codes
        df[var] = df[var].replace(-1, np.nan)
    df['infrastructure'] = df[existing_infra].mean(axis=1, skipna=True)

# 2. Create nat_res
res_vars = ['cm_res_limestone', 'cm_res_marble', 'cm_res_sand', 'cm_res_forest', 
            'cm_res_ebony', 'cm_res_exwood', 'cm_res_gold', 'cm_res_charcoal', 
            'cm_res_graphite', 'cm_res_stones', 'cm_res_mercury', 'cm_res_fishing', 
            'cm_res_salt', 'cm_res_nat_gas']

# Make sure res_vars are numeric before summing
for var in res_vars:
    if var in df.columns and df[var].dtype.name == 'category':
        df[var] = df[var].cat.codes

df['nat_res'] = df[res_vars].sum(axis=1, skipna=True) / 10

# 3. Create community averages - ensure variables are numeric first
for var in ['ethn_macua', 'ethn_maconde', 'educ_3']:
    if var in df.columns:
        # Convert to numeric if it's categorical
        if df[var].dtype.name == 'category':
            df[var] = df[var].cat.codes
        df[f'm{var}'] = df.groupby(['year', 'ae_id'], observed=True)[var].transform('mean')

# NOW filter to 2017
df = df[df['year'] == 2017].copy()

# =====================================
# Table 3 Column 7: Matching grants contribution
# =====================================

# Create treatment variables
df['tc1'] = df['tr_1'] 
df['tc2'] = df['tr_2'] + df['tr_3']

# Convert age to numeric if needed
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age2'] = df['age'] ** 2

# Convert gender if it's a string
if 'gender' in df.columns and df['gender'].dtype == 'object':
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Define dependent variable 
y_col7 = df['h31_conf'] + 1  

# Main treatment variable
main_treatment = 'tc2'

# Define all controls
all_controls = ['gender', 'age', 'age2', 'educ_2', 'educ_3', 'rel_muslim', 
                'ethn_macua', 'ethn_maconde', 'hh_size', 'a16', 'married', 'sub_farmer',
                'nat_res', 'num_tables_14', 'distpalma',
                'methn_macua', 'methn_maconde', 'meduc_3',
                'district1', 'district2', 'district3', 'district4', 'district5',
                'district6', 'district7', 'district8', 'district9', 'district10',
                'strata_rnd2', 'strata_rnd3']

# Add infrastructure if it exists
if 'infrastructure' in df.columns:
    all_controls.insert(13, 'infrastructure')

# Only use controls that exist
available_controls = [var for var in all_controls if var in df.columns]

# Filter out rows with missing dependent variable
valid_idx = ~df['h31_conf'].isna()
df_filtered = df[valid_idx].copy()
y_col7_filtered = df_filtered['h31_conf'] + 1

# Prepare X matrix
X_col7 = df_filtered[[main_treatment] + available_controls].copy()

# Convert any remaining categorical variables in X
for col in X_col7.columns:
    if X_col7[col].dtype.name == 'category':
        X_col7[col] = X_col7[col].cat.codes

# Drop missing values
X_col7 = X_col7.dropna()

y_col7_filtered = y_col7_filtered[X_col7.index]
cluster_col7 = df_filtered.loc[X_col7.index, 'ae_id']

# Add constant
X_col7 = sm.add_constant(X_col7, prepend=False)

# Ensure all data is numeric
X_col7 = X_col7.astype(float)

# METADATA
metadata_col7 = {
    'paper_id': '035',
    'table_id': '3',
    'panel_identifier': '7',
    'model_type': 'log-linear',
    'comments': 'Table 3 Column 7: Matching grants contribution log(h31_conf+1) with all controls'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col7, 
    y=y_col7_filtered, 
    X=X_col7, 
    interest='tc2', 
    elasticity=False,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col7}},
)