"""
Industrial Espionage and Productivity - Table 2 Column 3 Replication
Authors: Albrecht Glitz and Erik Meyersson (2019)
Replicating Table 2, Column 3
"""

import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
import os

from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '009/regdata_3_yes_.33_.06.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 2 Column 3: Change in log TFP gap
# Outcome = [ln(A_W,t+3) - ln(A_W,t)] - [ln(A_E,t+3) - ln(A_E,t)]
# = 3-year growth West TFP - 3-year growth East TFP
# =====================================

# Filter for observations with all necessary variables
df = df.dropna(subset=['inf_gva', 'diff_patents_gva', 'c3difflnTFP', 'weight_workers'])

# Filter for positive values
df = df.reset_index()

# Main treatment variable (Espionage indicator)
main_treatment = 'inf_gva'

# Controls - patent gap and lagged TFP gap, plus fixed effects
control_set = ['year', 'branch']

# =====================================
# PREPARE DATA WITHOUT MANUAL WEIGHTING
# =====================================

# Prepare y (outcome variable)
y_col3 = np.exp(df['c3difflnTFP'])

branch_columns = [column for column in df.columns if column.startswith('br_')]
yd_columns = [column for column in df.columns if column.startswith('yd_')]

# Prepare X (independent variables with constant)
X_col3 = sm.add_constant(df[[main_treatment] + branch_columns + yd_columns], prepend=False)

# Extract weights
weights_col3 = df['weight_workers']

# Cluster variable
cluster_col3 = df['branch']

# METADATA FOR THIS RESULT
metadata_col3 = {
    'paper_id': '009',
    'table_id': '2',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'Table 2 Column 3: Change in log TFP gap = ln(FRG_growth/GDR_growth) with patent gap and lagged TFP gap controls, weighted by workers'
}

# RUN THE REPLICATION with weights parameter
replicate(
    metadata=metadata_col3,
    y=y_col3,
    X=X_col3,
    interest=main_treatment,
    elasticity=False,
    # fe=['year', 'branch'],
    weights=weights_col3,  # Pass weights directly to the function
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col3}},
)
