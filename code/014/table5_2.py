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
filepath = os.path.join(INPUT_DATA_DIR, '014/week_panel.dta')
df = pd.read_stata(filepath)

# Filter for CAN==1
df = df[df['CAN'] == 1].copy()

# Drop rows with missing values for key variables before creating lags
df = df.dropna(subset=['ACLED_conflict', 'ln_ACLED_conflict', 'post_qualification', 'id', 'week', 'month_calendar'])

# Handle zeros: Add 1 to ACLED_conflict before log transformation
df['ACLED_conflict_plus1'] = df['ACLED_conflict'] + 1

# Sort by panel structure before creating lags
df = df.sort_values(['id', 'week'])

# Create lagged variables for ln_ACLED_conflict
df['L1_ln_ACLED_conflict'] = df.groupby('id')['ln_ACLED_conflict'].shift(1)
df['L2_ln_ACLED_conflict'] = df.groupby('id')['ln_ACLED_conflict'].shift(2)
df['L3_ln_ACLED_conflict'] = df.groupby('id')['ln_ACLED_conflict'].shift(3)
df['L4_ln_ACLED_conflict'] = df.groupby('id')['ln_ACLED_conflict'].shift(4)

# Drop rows with missing lagged values
df = df.dropna(subset=['L1_ln_ACLED_conflict', 'L2_ln_ACLED_conflict', 
                       'L3_ln_ACLED_conflict', 'L4_ln_ACLED_conflict'])

# =====================================
# Table 5 Column 2: ACLED conflict with lagged dependent variables
# =====================================

# Define dependent variable (ACLED_conflict + 1 in levels since replicate function takes log internally)
y_col2 = df['ACLED_conflict_plus1'].values

# Main treatment variable
main_treatment = 'post_qualification'

# Define lagged variables
lagged_vars = ['L1_ln_ACLED_conflict', 'L2_ln_ACLED_conflict', 
               'L3_ln_ACLED_conflict', 'L4_ln_ACLED_conflict']

# Define fixed effect variables
control_set = ['id', 'week', 'month_calendar']

# Prepare X matrix - treatment variable first, then lags, then FE controls
X_col2 = sm.add_constant(df[[main_treatment] + lagged_vars + control_set], prepend=False)

# Define cluster variable
cluster_col2 = df['id'].values

# METADATA FOR THIS RESULT
metadata_col2 = {
    'paper_id': '014',
    'table_id': '5',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comments': 'Table 5 Column 2: log(ACLED conflict + 1) on post_qualification with 4 lags and id, week, month_calendar FE'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col2, 
    y=y_col2, 
    X=X_col2, 
    interest='post_qualification', 
    elasticity=False, 
    fe=['id', 'week', 'month_calendar'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)

