#in progress 
import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '014/week_panel.dta')
df = pd.read_stata(filepath)

# Filter sample and set panel order (xtset id week)
df = df[df['CAN'] == 1].sort_values(['id', 'week']).copy()

# Create L1–L4 of the DV within id
for k in range(1, 5):
    df[f'ln_ACLED_conflict_L{k}'] = df.groupby('id')['ln_ACLED_conflict'].shift(k)

# Build the lagged sample for Column (2)
df_col2 = df.dropna(subset=['ln_ACLED_conflict',
                            'post_qualification',
                            'ln_ACLED_conflict_L1',
                            'ln_ACLED_conflict_L2',
                            'ln_ACLED_conflict_L3',
                            'ln_ACLED_conflict_L4']).copy()

# Main treatment and controls (treatment first)
main_treatment = 'post_qualification'
control_set = ['ln_ACLED_conflict_L1',
               'ln_ACLED_conflict_L2',
               'ln_ACLED_conflict_L3',
               'ln_ACLED_conflict_L4']

# Dependent and regressors
y_col2 = df_col2['ln_ACLED_conflict']
X_col2 = sm.add_constant(df_col2[[main_treatment] + control_set], prepend=False)

# Cluster groups
cluster_id2 = df_col2['id']

# Metadata
metadata_col2 = {
    'paper_id': '014',
    'table_id': '5',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comment': 'Column (2) of Table 5.2: Main treatment with lagged dependent variables as controls'
}

# Run: equivalent to Stata reghdfe with FE and vce(cluster id)
res_col2 = replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest=main_treatment,
    elasticity=False,
    fe=['id', 'week'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_id2}},
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)
