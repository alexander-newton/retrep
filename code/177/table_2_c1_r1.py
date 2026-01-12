import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
from replication import replicate

## This script replicates one of the main results from Agersnap and Zidar (2016)
## to calculate the elasticity that PolicyImpact.org quotes for their paper, we need to compute 4 elasticities:
## e^R is the elasticity they quote (think of this as a diff in diff)
## e^R=e^CG-e^N
## e^CG=e^CG_post-e^CG_pre
## e^N=e^N_post-e^N_pre
## So this script does 4 replications (e^CG_post, e^CG_pre, e^N_post, e^N_pre)
## to compute the standard error of e^R we need a matrix, which I don't have here.

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

df = df.set_index(['fips', 'year'])

#### POST REGRESSION (cg_post) ###

# Main treatment variable for 0-10 years (11-year change)
main_treatment_post = 'year100_d11log_ntr'

# Control variables: leads and lags
lagged_vars_post = ['year111_d3log_ntr', 'year114_d3log_ntr', 'year117_d3log_ntr',
                    'year91_d3log_ntr', 'year94_d3log_ntr', 'year97_d3log_ntr']

X_col1_post = sm.add_constant(df[[main_treatment_post] + lagged_vars_post], prepend=False)

# Define cluster variable
cluster_col1 = df.index.get_level_values('fips').values

# METADATA FOR THIS RESULT
metadata_post = {
    'paper_id': '177',
    'table_id': '2',
    'panel_identifier': '1',
    'model_type': 'log-log',
    'comments': 'Table 2 Column 1 Row 1: Baseline, 0-10 years, total elasticity, POST'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_post, 
    y=y_col1, 
    X=X_col1_post, 
    interest='year100_d11log_ntr', 
    elasticity=True, 
    fe=['fips', 'year'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
    # output=True, output_dir=OUTPUT_DIR, replicated=True
)


#### PRE REGRESSION (cg_pre) ###

# Main treatment variable (still year97 for pre-period)
main_treatment_pre = 'year97_d3log_ntr'

# Control variables for pre-period
lagged_vars_pre = ['year88_d3log_ntr', 'year91_d3log_ntr', 'year94_d3log_ntr',
                   'year100_d3log_ntr', 'year103_d3log_ntr', 'year106_d3log_ntr']

X_col1_pre = sm.add_constant(df[[main_treatment_pre] + lagged_vars_pre], prepend=False)

# METADATA FOR THIS RESULT
metadata_pre = {
    'paper_id': '177',
    'table_id': '2',
    'panel_identifier': '1',
    'model_type': 'log-log',
    'comments': 'Table 2 Column 1 Row 1: Baseline, 0-10 years, total elasticity, PRE'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_pre, 
    y=y_col1, 
    X=X_col1_pre, 
    interest='year97_d3log_ntr', 
    elasticity=True, 
    fe=['fips', 'year'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
    # output=True, output_dir=OUTPUT_DIR, replicated=True
)

## To get ε^CG elasticity: POST - PRE = 3.39 (1.01)

#### MIGRATION ELASTICITY (ε^N) - 0-10 YEARS ####

# Define dependent variable for migration
y_migration = np.exp(df['log_n_tmix'])

#### POST REGRESSION (mig_post) ###

# Main treatment variable for 0-10 years (11-year change)
main_treatment_post = 'year100_d11log_ntr'

# Control variables: leads and lags (same as CG regression)
lagged_vars_post = ['year111_d3log_ntr', 'year114_d3log_ntr', 'year117_d3log_ntr',
                    'year91_d3log_ntr', 'year94_d3log_ntr', 'year97_d3log_ntr']

X_mig_post = sm.add_constant(df[[main_treatment_post] + lagged_vars_post], prepend=False)

# Define cluster variable
cluster_mig = df.index.get_level_values('fips').values

# METADATA FOR THIS RESULT
metadata_mig_post = {
    'paper_id': '177',
    'table_id': '2',
    'panel_identifier': '1',
    'model_type': 'log-log',
    'comments': 'Table 2 Column 1 Row 1: Baseline, 0-10 years, migration elasticity ε^N, POST'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_mig_post, 
    y=y_migration, 
    X=X_mig_post, 
    interest='year100_d11log_ntr', 
    elasticity=True, 
    fe=['fips', 'year'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_mig}},
    # output=True, output_dir=OUTPUT_DIR, replicated=True
)


#### PRE REGRESSION (mig_pre) ###

# Main treatment variable (still year97 for pre-period)
main_treatment_pre = 'year97_d3log_ntr'

# Control variables for pre-period
lagged_vars_pre = ['year88_d3log_ntr', 'year91_d3log_ntr', 'year94_d3log_ntr',
                   'year100_d3log_ntr', 'year103_d3log_ntr', 'year106_d3log_ntr']

X_mig_pre = sm.add_constant(df[[main_treatment_pre] + lagged_vars_pre], prepend=False)

# METADATA FOR THIS RESULT
metadata_mig_pre = {
    'paper_id': '177',
    'table_id': '2',
    'panel_identifier': '1',
    'model_type': 'log-log',
    'comments': 'Table 2 Column 1 Row 1: Baseline, 0-10 years, migration elasticity ε^N, PRE'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_mig_pre, 
    y=y_migration, 
    X=X_mig_pre, 
    interest='year97_d3log_ntr', 
    elasticity=True, 
    fe=['fips', 'year'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_mig}},
    # output=True, output_dir=OUTPUT_DIR, replicated=True
)

## To get ε^N elasticity: POST - PRE ≈ 1.52
## Then calculate ε^CG - ε^N = 3.39 - 1.52 = 1.87 (which should match the table)
