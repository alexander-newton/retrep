import yaml
import os
import pandas as pd
import numpy as np

import sys
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '177/cgt_rates_inc_panel_bystate.dta')
df = pd.read_stata(filepath)

# Define dependent variable
y_col1 = np.exp(df['log_inc_kg_r'])

# Main treatment variable
main_treatment_post = 'year100_d3log_ntr'

df = df.set_index(['fips', 'year'])

control_set = ['fips', 'year']

# Regression 1: year100 (0-2 year period)
lagged_vars_post = ['year91_d3log_ntr', 'year94_d3log_ntr', 'year97_d3log_ntr', 
                'year103_d3log_ntr', 'year106_d3log_ntr', 'year109_d3log_ntr']

X_col1_post = sm.add_constant(df[[main_treatment_post] + lagged_vars_post + control_set], prepend=False)

# Define cluster variable
cluster_col1row2 = df['fips'].values

# METADATA FOR THIS RESULT
metadata_col1row2 = {
    'paper_id': '177',
    'table_id': '2',
    'panel_identifier': '1',
    'model_type': 'log-log',
    'comments': 'Table 2 Column 1 Row 2: Baseline, 0-2 years, total elasticity, POST'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col1row2, 
    y=y_col1, 
    X=X_col1_post, 
    interest='year100_d3log_ntr', 
    elasticity=True, 
    fe=['fips', 'year'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1row2}},
    # output=True, output_dir=OUTPUT_DIR, replicated=True
)

# estimated (post) elasticity is -.54068  from the authors


#### PRE ###

# Main treatment variable
main_treatment_pre = 'year97_d3log_ntr'

lagged_vars_pre = ['year88_d3log_ntr', 'year91_d3log_ntr', 'year94_d3log_ntr', 'year100_d3log_ntr', 'year103_d3log_ntr', 'year106_d3log_ntr']

X_col1_pre = sm.add_constant(df[[main_treatment_pre] + lagged_vars_pre + control_set], prepend=False)

# Define cluster variable
cluster_col1row2 = df['fips'].values

# METADATA FOR THIS RESULT
metadata_col1row2 = {
    'paper_id': '177',
    'table_id': '2',
    'panel_identifier': '1',
    'model_type': 'log-log',
    'comments': 'Table 2 Column 1 Row 2: Baseline, 0-2 years, total elasticity, PRE'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col1row2, 
    y=y_col1, 
    X=X_col1_pre, 
    interest='year97_d3log_ntr', 
    elasticity=True, 
    fe=['fips', 'year'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1row2}},
    # output=True, output_dir=OUTPUT_DIR, replicated=True
)

# estimated (pre) elasticity is -3.860793 from the authors

## to get elasticity need to subtract: POST-PRE