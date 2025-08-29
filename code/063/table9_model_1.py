import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

# Add parent directory to path for replication module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# =====================================
# LOAD DATA
# =====================================
filepath = os.path.join(INPUT_DATA_DIR, '063/firm_data_model.dta')
df = pd.read_stata(filepath)

# =====================================
# CREATE VARIABLES (following Stata code)
# =====================================
# Sort by firm and time to ensure proper lagging
df = df.sort_values(['ffirm', 'time'])

# Create percentiles for A2f normalization
Ap50 = df['A2f'].quantile(0.50)
Ap90 = df['A2f'].quantile(0.90)
df['A2f'] = df['A2f'] / (Ap90 - Ap50)

# Standardize A2m
df['AIstd'] = (df['A2m'] - df['A2m'].mean()) / df['A2m'].std()
df['A2m'] = df['AIstd']

# Create lagged variables
df['LA2m'] = df.groupby('ffirm')['A2m'].shift(1)
df['L_A2f'] = df.groupby('ffirm')['A2f'].shift(1)
df['L_mA2f'] = df.groupby('ffirm')['mA2f'].shift(1)
df['L_logY'] = df.groupby('ffirm')['logY'].shift(1)
df['L2_logY'] = df.groupby('ffirm')['logY'].shift(2)

# Create forward variables for logY
df['F_logY'] = df.groupby('ffirm')['logY'].shift(-1)

# Create interaction term
# First, mean-center LA2m
mLA2m = df['LA2m'].mean()
df['qXLSA2i'] = df['q_dum2'] * (df['LA2m'] - mLA2m)

# Create outcome variables (forward differences)
df['F0DlogY'] = df['logY'] - df['L_logY']
df['F1DlogY'] = df['F_logY'] - df['L_logY']

# =====================================
# Table 9 MODEL Column 1: F0DlogY (t=0)
# Based on: reg F0DlogY L.mA2f LA2m qXLSA2i LlogK q_dum2 L.logY, cluster(ffirm)
# =====================================

# Remove rows with missing values for key variables
required_vars = ['F0DlogY', 'L_mA2f', 'LA2m', 'qXLSA2i', 'LlogK', 'q_dum2', 'L_logY', 'ffirm']
df_clean = df.dropna(subset=required_vars)

# Convert F0DlogY to levels if needed (exponential transformation)
df_clean['F0DY'] = np.exp(df_clean['F0DlogY'])

# Prepare dependent variable (in levels)
y_col1 = df_clean['F0DY']

# Prepare independent variables
# Main variables of interest: LA2m and qXLSA2i (as shown in esttab keep())
# Control variables: L.mA2f, LlogK, q_dum2, L.logY
X_vars = ['LA2m', 'qXLSA2i', 'L_mA2f', 'LlogK', 'q_dum2', 'L_logY']
X_col1 = df_clean[X_vars].copy()

# Add constant (for regression)
X_col1 = sm.add_constant(X_col1, prepend=False)

# Cluster variable
cluster_col1 = df_clean['ffirm']

# =====================================
# METADATA FOR THIS RESULT
# =====================================
metadata_col1 = {
    'paper_id': '063',
    'table_id': '9',
    'panel_identifier': 'model_1',
    'model_type': 'log-linear',
    'comments': 'Table 9 (MODEL) Column 1: Forward difference of output at t=0 (in levels), clustered SE by ffirm'
}

# =====================================
# RUN THE REPLICATION
# =====================================
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest=['LA2m'],  # Variables of interest from esttab keep()
    elasticity=False,
    fe=None,  # No fixed effects absorbed in this model specification
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
 
)