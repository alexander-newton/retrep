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
filepath = os.path.join(INPUT_DATA_DIR, '063/Firm_Profit_Innovation.dta')
df = pd.read_stata(filepath)

# =====================================
# CREATE LAGGED VARIABLES
# =====================================
df = df.sort_values(['permno', 'year'])

# Create lagged logY (L.logY in Stata)
df['L_logY'] = df.groupby('permno')['logY'].shift(1)

# =====================================
# Table 9 data Column 1: F0DlogY (t=0)
# Based on: areg F0DlogY LlogK i.year LSA2i qXLSA2i q_dum2 LSA2f L.logY, absorb(indcd) cluster(permno)
# =====================================

# Remove rows with missing values for key variables (including the lagged variable)
required_vars = ['F0DlogY', 'LlogK', 'LSA2i', 'qXLSA2i', 'q_dum2', 'LSA2f', 'L_logY', 'year', 'indcd', 'permno']
df_clean = df.dropna(subset=required_vars)

df_clean['F0DY'] = np.exp(df_clean['F0DlogY'])

# Prepare dependent variable
y_col1 = df_clean['F0DY']
#df['loans'] = np.exp(df['logloans'])
# Prepare independent variables
# Main variables of interest: LSA2i and qXLSA2i
# Control variables: LlogK, q_dum2, LSA2f, L_logY (lagged logY)
# Note: Year is handled as FE, not in X matrix
X_vars = ['LSA2i', 'qXLSA2i','LlogK', 'q_dum2', 'LSA2f', 'L_logY','indcd', 'year']
X_col1 = df_clean[X_vars].copy()

# Add constant (for regression)
X_col1 = sm.add_constant(X_col1, prepend=False)

# Fixed effects to absorb: both industry (indcd) and year
# Note: In Stata, i.year creates year FE and absorb(indcd) absorbs industry FE
fe_vars = ['indcd', 'year']

# Cluster variable
cluster_col1 = df_clean['permno']

# =====================================
# METADATA FOR THIS RESULT
# =====================================
metadata_col1 = {
    'paper_id': '063',
    'table_id': '9',
    'panel_identifier': 'data_1',
    'model_type': 'log-linear',
    'comments': 'Table 9 (data) Column 1: Forward difference of log output at t=0 with industry and year FE, clustered SE by permno'
}

# =====================================
# RUN THE REPLICATION
# =====================================
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest=['LSA2i'],
    elasticity=False,
    fe=fe_vars,  # Both industry and year fixed effects
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
    output=True, output_dir=OUTPUT_DIR, replicated=True

)