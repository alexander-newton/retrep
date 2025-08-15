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
# APPLY WEIGHTS MANUALLY (WLS transformation)
# =====================================

# For weighted least squares, multiply both y and X by sqrt(weights)
weights_sqrt = np.sqrt(df['weight_workers'])

# Weight the dependent variable
y_col6 = df['gvapc_gap_change_ratio'] * weights_sqrt

# Weight the independent variables
X_weighted = df[[main_treatment] + control_set].copy()
for col in X_weighted.columns:
    X_weighted[col] = X_weighted[col] * weights_sqrt

# Add constant AFTER weighting
X_col6 = sm.add_constant(X_weighted, prepend=False)

# Cluster variable (not weighted)
cluster_col6 = df['branch']

# METADATA FOR THIS RESULT
metadata_col6 = {
    'paper_id': '009',
    'table_id': '2',
    'panel_identifier': '6',
    'model_type': 'log-linear',
    'comments': 'Table 2 Column 6: Change in log output per worker gap = ln(GVAPC_FRG_growth/GVAPC_GDR_growth) with patent gap and lagged output/worker gap controls, weighted by workers'
}


replicate(
    metadata=metadata_col6,
    y=y_col6,
    X=X_col6,
    interest='inf_gva',
    elasticity=False,
    fe=['year', 'branch'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col6}},
)