"""
Industrial Espionage and Productivity - Table 2 Column 6 Replication
Authors: Albrecht Glitz and Erik Meyersson (2019)
Replicating Table 2, Column 6 (log output per worker with lagged gap)
"""

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
filepath = os.path.join(INPUT_DATA_DIR, '009/regdata_3_yes_.33_.06.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 2 Column 6: Change in log output per worker gap
# Outcome = [ln(Y/L_W,t+3) - ln(Y/L_W,t)] - [ln(Y/L_E,t+3) - ln(Y/L_E,t)]
# = 3-year growth West output per worker - 3-year growth East output per worker
# =====================================

# Sort by branch and year
df = df.sort_values(['branch', 'year'])

# Create output per worker variables
# Based on the column names, we have:
# FRG_gva_labour_ratio_ = West German GVA per worker
# GDR_gva_labour_ratio_ = East German GVA per worker

# Create 3-year forward values for output per worker
df['FRG_gvapc_t3'] = df.groupby('branch')['FRG_gva_labour_ratio_'].shift(-3)
df['GDR_gvapc_t3'] = df.groupby('branch')['GDR_gva_labour_ratio_'].shift(-3)

# Calculate the growth ratios for each country
# West growth: FRG_gvapc_t3 / FRG_gva_labour_ratio_
# East growth: GDR_gvapc_t3 / GDR_gva_labour_ratio_
df['FRG_gvapc_growth'] = df['FRG_gvapc_t3'] / df['FRG_gva_labour_ratio_']
df['GDR_gvapc_growth'] = df['GDR_gvapc_t3'] / df['GDR_gva_labour_ratio_']

# The outcome in levels is the ratio of growth rates
# When logged: ln(FRG_growth/GDR_growth) = ln(FRG_growth) - ln(GDR_growth)
df['gvapc_gap_change_ratio'] = df['FRG_gvapc_growth'] / df['GDR_gvapc_growth']

# Filter for observations with all necessary variables
# Note: for Column 6, we use diffln_gvapc (the lagged output per worker gap) instead of difflnTFP
df = df.dropna(subset=['gvapc_gap_change_ratio', 'inf_gva', 'diff_patents_gva', 'diffln_gvapc', 'weight_workers'])

# Filter for positive values
df = df[(df['gvapc_gap_change_ratio'] > 0)]

# Main treatment variable (Espionage indicator)
main_treatment = 'inf_gva'

# Controls - patent gap and lagged output per worker gap, plus fixed effects
control_set = ['diff_patents_gva', 'diffln_gvapc', 'year', 'branch']

# =====================================
# PREPARE DATA WITHOUT MANUAL WEIGHTING
# =====================================

# Prepare y (outcome variable)
y_col6 = df['gvapc_gap_change_ratio']

# Prepare X (independent variables with constant)
X_col6 = sm.add_constant(df[[main_treatment] + control_set], prepend=False)

# Extract weights
weights_col6 = df['weight_workers']

# Cluster variable
cluster_col6 = df['branch']

# METADATA FOR THIS RESULT
metadata_col6 = {
    'paper_id': '009',
    'table_id': '2',
    'panel_identifier': '6',
    'model_type': 'log-linear',
    'comments': 'Table 2 Column 6: Change in log output per worker gap = ln(GVAPC_FRG_growth/GVAPC_GDR_growth) with patent gap and lagged output/worker gap controls, weighted by workers'
}

# RUN THE REPLICATION with weights parameter
replicate(
    metadata=metadata_col6,
    y=y_col6,
    X=X_col6,
    interest='inf_gva',
    elasticity=False,
    fe=['year', 'branch'],
    weights=weights_col6,  # Pass weights directly to the function
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col6}},
)