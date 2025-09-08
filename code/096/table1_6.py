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
filepath = os.path.join(INPUT_DATA_DIR, '096/Price_kgAll_Main_clean_0_980000_0_980000_50.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 1 Column 6: Kidney Beans Unit Values
# =====================================

# Create village identifier at locality level
df = df.sort_values(['state', 'muni', 'local', 'folio'])
df['village2'] = df.groupby(['state', 'muni', 'local']).ngroup()

# Generate log variables
df['lp_kidbeans'] = np.log(df['price_real_kidbeans'])
df['lq_kidbeans'] = np.log(df['KG_kidbeans'])

# Generate treatment variable
df['treat2'] = df['treat'].copy()
df.loc[(df['treat'] == 0) & (df['wave'].str.contains('2000|2003', na=False)), 'treat2'] = 1

# Create interaction term
df['lq_kidbeans_t2'] = df['lq_kidbeans'] * df['treat2']

# Remove missing values
df = df.dropna(subset=['lp_kidbeans', 'treat2', 'lq_kidbeans', 'lq_kidbeans_t2', 'wave', 'village2'])

# Dependent variable
y_col6 = df['price_real_kidbeans']

# Main treatment variable
main_treatment = 'treat2'

# Controls / FE
control_set = ['lq_kidbeans', 'lq_kidbeans_t2', 'wave']

# X matrix: treatment first (for easy identification)
X_col6 = sm.add_constant(df[[main_treatment, 'lq_kidbeans', 'lq_kidbeans_t2', 'wave']], prepend=False)

# Cluster variable
cluster_col6 = df['village2']

# METADATA FOR THIS RESULT
metadata_col6 = {
    'paper_id': '096',
    'table_id': '1',
    'panel_identifier': '6',
    'model_type': 'log-linear',
    'comments': 'Table 1 Column 6: Kidney Beans Unit Values with treatment, log(q), interaction, and wave FE'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col6,
    y=y_col6,
    X=X_col6,
    interest='treat2',
    elasticity=False,
    fe=['wave'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col6}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)