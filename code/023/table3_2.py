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
filepath = os.path.join(INPUT_DATA_DIR, '023/CityShape_Main.dta')
df = pd.read_stata(filepath)

# Filter for sample: 351 cities
df = df[df['insample_IV_5010'] == 1].copy()

# Reshape data from long to wide format
df_1950 = df[df['year'] == 1950].copy()
df_2010 = df[df['year'] == 2010].copy()

# Merge the two years
df_wide = pd.merge(df_1950, df_2010, on='id', suffixes=('_1950', '_2010'))

# Calculate long differences
# Use ratio so internal log() becomes Δ log:
df_wide['TOTAL_pop_all_ratio'] = df_wide['TOTAL_pop_all_2010'] / df_wide['TOTAL_pop_all_1950']

# (safety) drop invalid denominators or nonpositive ratios
df_wide = df_wide[df_wide['TOTAL_pop_all_1950'] > 0]
df_wide = df_wide[df_wide['TOTAL_pop_all_ratio'] > 0]

# Independent variables: changes in shape and log area
df_wide['disconnect_km_diff'] = df_wide['disconnect_km_2010'] - df_wide['disconnect_km_1950']
df_wide['log_area_polyg_km_diff'] = df_wide['log_area_polyg_km_2010'] - df_wide['log_area_polyg_km_1950']

# Drop missing values
sample_vars = ['TOTAL_pop_all_diff', 'disconnect_km_diff', 'log_area_polyg_km_diff']
df_wide = df_wide.dropna(subset=sample_vars)

# Dependent variable
y_col2 = df_wide['TOTAL_pop_all_ratio']

# X matrix
X_vars = ['disconnect_km_diff', 'log_area_polyg_km_diff']
X_col2 = sm.add_constant(df_wide[X_vars], prepend=False)

# Cluster variable
cluster_col2 = df_wide['id']

# METADATA
metadata_col2 = {
    'paper_id': '023',
    'table_id': '3',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comments': 'Table 3 Panel A Column 2: Population Ratio on Disconnect and Log Area with OLS'
}

# RUN REPLICATION
replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest='disconnect_km_diff',
    elasticity=False,
    fe=None,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
    output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True
)

