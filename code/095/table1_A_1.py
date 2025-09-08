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
# Adjust the subfolder/file name if your repo uses a different path
# ===============================
filepath = os.path.join(INPUT_DATA_DIR, '095/MergedDataset.dta')
df = pd.read_stata(filepath).dropna(subset=['delta_tfp', 'dummy_20', 't', 'sic_group'])

# ===============================
# Table 1 Panel A Column 1
# Δ log TFP on Jaccard>20% dummy, Time FE only, All industries, unweighted
# ===============================

# Sample: all industries (Panel A)
# If your dataset uses 'all'==1 to denote inclusion in full sample:
if 'all' in df.columns:
    df = df[df['all'] == 1]

# Dependent variable: five-year change in log TFP
y_col1 = np.exp(df['delta_tfp'])

# Main treatment variable: Jaccard distance above 20th pct dummy
main_treatment = 'dummy_20'

# Fixed effects: time only (i.t in the Stata file)
# Keep 't' in X (replicate() may also build FE internally; mirroring your template style)
control_set = ['t']

# X matrix: treatment first
X_col1 = sm.add_constant(df[[main_treatment] + control_set], prepend=False)

# Cluster variable (clustered at industry level)
cluster_col1 = df['sic_group']

# METADATA
metadata_col1 = {
    'paper_id': '095',
    'table_id': '1',
    'panel_identifier': 'A_1',
    'model_type': 'log-linear',
    'comments': 'Table 1 Panel A Column 1: Δlog TFP on Jaccard>20% dummy with time FE only; all industries; unweighted; cluster by industry (sic_group)'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest=main_treatment,
    elasticity=False,
    fe=['t'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
