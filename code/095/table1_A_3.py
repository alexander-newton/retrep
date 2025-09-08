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

# ===============================
# LOAD DATA
# ===============================
filepath = os.path.join(INPUT_DATA_DIR, '095/MergedDataset.dta')
df = pd.read_stata(filepath).dropna(subset=['delta_tfp', 'dummy_20', 't', 'sic_group'])

# Sample: all industries (Panel A)
if 'all' in df.columns:
    df = df[df['all'] == 1]

# ===============================
# CREATE LAGGED DEPENDENT VARIABLE
# Equivalent of Stata's L.delta_tfp
# ===============================
df = df.sort_values(['sic_group', 't'])
df['lag_delta_tfp'] = df.groupby('sic_group')['delta_tfp'].shift(1)

# Drop missing lag values
df = df.dropna(subset=['lag_delta_tfp'])

# ===============================
# Table 1 Panel A Column 3
# Δ log TFP on Jaccard>20% dummy, Lagged Δ log TFP, Time + Industry FE
# ===============================

# Dependent variable
y_col3 = np.exp(df['delta_tfp'])

# Main treatment variable
main_treatment = 'dummy_20'

# Controls: include lagged Δ log TFP + time FE
control_set = ['t', 'sic_group','lag_delta_tfp']

# X matrix: treatment first
X_col3 = sm.add_constant(df[[main_treatment] + control_set], prepend=False)

# Cluster variable
cluster_col3 = df['sic_group']

# METADATA
metadata_col3 = {
    'paper_id': '095',
    'table_id': '1',
    'panel_identifier': 'A_3',
    'model_type': 'log-linear',
    'comments': 'Table 1 Panel A Column 3: Δlog TFP on Jaccard>20% dummy with lagged Δlog TFP, time FE, and industry FE; all industries; unweighted; cluster by industry (sic_group).'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col3,
    y=y_col3,
    X=X_col3,
    interest=main_treatment,
    elasticity=False,
    fe=['t', 'sic_group'],  # same as Col 2, but add lagged DV
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col3}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
