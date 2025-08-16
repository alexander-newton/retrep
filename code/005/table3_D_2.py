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

# =====================================
# LOAD 1936 DATA
# =====================================
filepath = os.path.join(INPUT_DATA_DIR, '005/houses_1936_final.dta')
df_1936 = pd.read_stata(filepath).copy()

# =====================================
# DATA PREPARATION
# =====================================
# Note: 1936 data appears to already be in meters (no scaling needed)
# Create running variable (negative outside BSP, positive inside)
df_1936['dist_2'] = df_1936['dist_netw'].copy()
df_1936.loc[df_1936['broad'] == 0, 'dist_2'] = -df_1936['dist_netw']

# Create polynomial terms for consistency
df_1936['dist_netw2'] = df_1936['dist_netw'] ** 2

# =====================================
# DEFINE CONTROLS FOR PANEL D
# =====================================
# For panel D, controls are: distance to neighborhood centroid,
# distance to closest: square, theater, pub, church, and bank
controls_1936 = [
    'dist_cent',    # distance to neighborhood centroid
    'dist_square',  # distance to square
    'dist_thea',    # distance to theater
    'dist_pub',     # distance to pub
    'dist_church',  # distance to church
    'dist_bank'     # distance to bank
]

# =====================================
# DATA CLEANING
# =====================================
# Drop missing values - using rentals in levels
df_clean = df_1936.dropna(subset=['rentals', 'dist_2', 'block'] + controls_1936)

# =====================================
# OPTIMAL BANDWIDTH
# =====================================
# Based on the code, they use a fixed bandwidth of 0.373 for column 2
optimal_bw = 0.373  # 37.3 meters

# =====================================
# REPLICATION FRAMEWORK - PANEL D COLUMN 2
# =====================================
# Filter to optimal bandwidth
df_bw = df_clean[np.abs(df_clean['dist_2']) <= optimal_bw].copy()
df_bw['treatment_rd'] = (df_bw['dist_2'] > 0).astype(int)

# Calculate triangular kernel weights (matching rdrobust default)
rd_weights = 1 - np.abs(df_bw['dist_2'].values) / optimal_bw

# Prepare variables - USE RENTALS IN LEVELS
y_d2 = df_bw['rentals'].values  # 1936 rental values in LEVELS
X_d2 = sm.add_constant(df_bw[['treatment_rd', 'dist_2', 'dist_netw2'] + controls_1936], prepend=False)
cluster_d2 = df_bw['block'].values

# Metadata for Panel D Column 2
metadata_d2 = {
    'paper_id': '005',
    'table_id': '3',
    'panel_identifier': 'D_2',
    'model_type': 'log-linear',
    'comments': f'Table 3 Panel D Column 2: 1936 rentals, LLR with controls (Panel D control set); Bandwidth: {optimal_bw*100:.2f}m'
}

# Run replication
replicate(
    metadata=metadata_d2,
    y=y_d2,  # rentals in LEVELS
    X=X_d2,
    interest='treatment_rd',
    elasticity=False,
    fe=None,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_d2}},
    weights=rd_weights
)