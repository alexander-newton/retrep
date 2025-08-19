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
# Dependent variable: change in population (in levels)
df_wide['TOTAL_pop_all_ratio'] = df_wide['TOTAL_pop_all_2010'] / df_wide['TOTAL_pop_all_1950']

# Endogenous variables: changes in shape (NOT normalized) and log area
df_wide['disconnect_km_diff'] = df_wide['disconnect_km_2010'] - df_wide['disconnect_km_1950']
df_wide['log_area_polyg_km_diff'] = df_wide['log_area_polyg_km_2010'] - df_wide['log_area_polyg_km_1950']

# Instruments: changes in potential shape and log projected population
df_wide['r1_relev_disconnect_cls_km_diff'] = df_wide['r1_relev_disconnect_cls_km_2010'] - df_wide['r1_relev_disconnect_cls_km_1950']
df_wide['log_projected_pop_diff'] = df_wide['log_projected_pop_2010'] - df_wide['log_projected_pop_1950']

# Drop missing values
sample_vars = ['TOTAL_pop_all_ratio', 'disconnect_km_diff', 'log_area_polyg_km_diff',
               'r1_relev_disconnect_cls_km_diff', 'log_projected_pop_diff']
df_wide = df_wide.dropna(subset=sample_vars)

# Dependent variable
y_col1 = df_wide['TOTAL_pop_all_ratio']

# X matrix (endogenous variables that will be instrumented)
X_vars = ['disconnect_km_diff', 'log_area_polyg_km_diff']
X_col1 = sm.add_constant(df_wide[X_vars], prepend=False)

# Z matrix (instruments)
Z_vars = ['r1_relev_disconnect_cls_km_diff', 'log_projected_pop_diff']
Z_col1 = sm.add_constant(df_wide[Z_vars], prepend=False)

# Cluster variable
cluster_col1 = df_wide['id']

# METADATA
metadata_col1 = {
    'paper_id': '023',
    'table_id': '3', 
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 3 Panel A Column 1: Population Ratio on Disconnect and Log Area with IVs'
}

# RUN REPLICATION - IV estimation
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest='disconnect_km_diff',
    endog_x=['disconnect_km_diff', 'log_area_polyg_km_diff'],  # Both are endogenous
    z=Z_col1,  # Instruments
    elasticity=False,
    fe=None,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
    kwargs_estimator={'estimator_type': 'iv'},  # Specify IV estimation
    output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True
)