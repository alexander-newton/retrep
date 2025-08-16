import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Load data
df = pd.read_stata(os.path.join(INPUT_DATA_DIR, '008/pmgsy_working_aer_mainsample.dta'))

# Apply filters exactly as in Stata
# 1. Main sample
df = df[df['mainsample'] == 1].copy()

# 2. No bad firms filter: inrange(ec13_emp_share, 0, 1)
df = df[(df['ec13_emp_share'] >= 0) & (df['ec13_emp_share'] <= 1)].copy()

# Define controls exactly as in settings.do
controls = ['primary_school', 'med_center', 'elect', 'tdist', 'irr_share', 'ln_land',
            'pc01_lit_share', 'pc01_sc_share', 'bpl_landed_share',
            'bpl_inc_source_sub_share', 'bpl_inc_250plus']

# RD polynomial terms
rd_terms = ['left', 'right']

# All RHS variables (excluding district FE for now)
all_controls = rd_terms + controls

# Column 1: Total employment (all firms) - USE LEVEL VARIABLE for log transformation
y_var = 'ec13_emp_all'  # Level variable, NOT ec13_emp_all_ln

# Weight column: kernel_tri_ik (triangular kernel at IK bandwidth)
weight_col = 'kernel_tri_ik'

# Prepare variables list for cleaning
vars_needed = [y_var, 'r2012', 't'] + all_controls + ['vhg_dist_id']
if weight_col in df.columns:
    vars_needed.append(weight_col)

# Remove rows with missing values
df_clean = df[vars_needed].dropna()

# Remove zero or negative values since log will be taken
df_clean = df_clean[df_clean[y_var] > 0]

# Also ensure positive weights if using them
if weight_col in df.columns:
    df_clean = df_clean[df_clean[weight_col] > 0]

print(f"Final sample size: {len(df_clean)}")  # Should be 10,678

# Prepare variables
y = df_clean[y_var].values.astype(float)

# Build X matrix with district dummies
district_dummies = pd.get_dummies(df_clean['vhg_dist_id'], prefix='dist', drop_first=True)

# Combine all variables: r2012 first, then RD terms, then controls, then district FEs, then constant
X = pd.concat([
    df_clean[['r2012']],
    df_clean[all_controls],
    district_dummies
], axis=1)
X = sm.add_constant(X, prepend=False)

# Convert to numpy array with float type
X = X.values.astype(float)

# Get weights if available
weights = df_clean[weight_col].values.astype(float) if weight_col in df_clean.columns else None

# Instrument - convert to float numpy array
z = df_clean[['t']].values.astype(float)

# Metadata
metadata = {
    'paper_id': '008',
    'table_id': '6',
    'panel_identifier': 'A_1',
    'model_type': 'log-linear',
    'comments': 'Table 6 Panel A Column 1: Log employment growth - Total (2SLS with controls and district FE)'
}

# Run replication
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=0,  # r2012 is at index 0
    weights=weights,
    endog_x=[0],
    z=z
)





