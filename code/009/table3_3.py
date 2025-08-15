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
# Table 2 Column 3: Change in log TFP gap
# Outcome = [ln(A_W,t+3) - ln(A_W,t)] - [ln(A_E,t+3) - ln(A_E,t)]
# = 3-year growth West TFP - 3-year growth East TFP
# =====================================

# Sort by branch and year
df = df.sort_values(['branch', 'year'])

# Create 3-year forward TFP levels for both West (FRG) and East (GDR) Germany
df['FRG_TFP_t3'] = df.groupby('branch')['FRG_TFP_'].shift(-3)
df['GDR_TFP_t3'] = df.groupby('branch')['GDR_TFP_'].shift(-3)

# Calculate the growth ratios for each country
# West growth: FRG_TFP_t3 / FRG_TFP_
# East growth: GDR_TFP_t3 / GDR_TFP_
df['FRG_growth'] = df['FRG_TFP_t3'] / df['FRG_TFP_']
df['GDR_growth'] = df['GDR_TFP_t3'] / df['GDR_TFP_']

# The outcome in levels is the ratio of growth rates
# When logged: ln(FRG_growth/GDR_growth) = ln(FRG_growth) - ln(GDR_growth)
# = [ln(FRG_t+3) - ln(FRG_t)] - [ln(GDR_t+3) - ln(GDR_t)]
df['gap_change_ratio'] = df['FRG_growth'] / df['GDR_growth']

# Filter for observations with all necessary variables
df = df.dropna(subset=['gap_change_ratio', 'inf_gva', 'diff_patents_gva', 'difflnTFP', 'weight_workers'])

# Filter for positive values
df = df[(df['gap_change_ratio'] > 0)]

# Main treatment variable (Espionage indicator)
main_treatment = 'inf_gva'

# Controls - patent gap and lagged TFP gap, plus fixed effects
control_set = ['diff_patents_gva', 'difflnTFP', 'year', 'branch']

# =====================================
# APPLY WEIGHTS MANUALLY (WLS transformation)
# =====================================

# For weighted least squares, multiply both y and X by sqrt(weights)
weights_sqrt = np.sqrt(df['weight_workers'])

# Weight the dependent variable
y_col3 = df['gap_change_ratio'] * weights_sqrt

# Weight the independent variables
X_weighted = df[[main_treatment] + control_set].copy()
for col in X_weighted.columns:
    X_weighted[col] = X_weighted[col] * weights_sqrt

# Add constant AFTER weighting
X_col3 = sm.add_constant(X_weighted, prepend=False)

# Cluster variable (not weighted)
cluster_col3 = df['branch']

# METADATA FOR THIS RESULT
metadata_col3 = {
    'paper_id': '009',
    'table_id': '2',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'Table 2 Column 3: Change in log TFP gap = ln(FRG_growth/GDR_growth) with patent gap and lagged TFP gap controls, weighted by workers'
}

# RUN THE REPLICATION with pre-weighted data
replicate(
    metadata=metadata_col3,
    y=y_col3,
    X=X_col3,
    interest='inf_gva',
    elasticity=False,
    fe=['year', 'branch'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col3}},
)