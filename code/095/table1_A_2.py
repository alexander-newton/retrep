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
# Table 1 Panel A Column 2
# Δ log TFP on Jaccard>20% dummy, Time + Industry FE, All industries, unweighted
# ===============================

# Dependent variable
y_col2 = np.exp(df['delta_tfp'])

# Main treatment variable
main_treatment = 'dummy_20'

# Controls / FE placeholders (we'll pass FE explicitly)
control_set = ['t', 'sic_group'] 

# X matrix: treatment first
X_col2 = sm.add_constant(df[[main_treatment] + control_set], prepend=False)

# Cluster variable
cluster_col2 = df['sic_group']

# METADATA
metadata_col2 = {
    'paper_id': '095',
    'table_id': '1',
    'panel_identifier': 'A_2',
    'model_type': 'log-linear',
    'comments': 'Table 1 Panel A Column 2: Δlog TFP on Jaccard>20% dummy with time and industry FE; all industries; unweighted; cluster by industry (sic_group).'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest=main_treatment,
    elasticity=False,
    fe=['t', 'sic_group'],  # add industry FE vs. Column 1
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
