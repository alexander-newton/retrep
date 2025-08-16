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
# LOAD 1894 DATA
# =====================================
filepath = os.path.join(INPUT_DATA_DIR, '005/Merged_1846_1894_data.dta')
df_1894 = pd.read_stata(filepath).copy()

# =====================================
# DATA PREPARATION
# =====================================
# Scale distance and create RD running variable
df_1894['dist_netw'] = df_1894['dist_netw'] / 100
df_1894['dist_netw2'] = df_1894['dist_netw'] ** 2

# Create running variable (negative outside BSP, positive inside)
df_1894['dist_2'] = df_1894['dist_netw'].copy()
df_1894.loc[df_1894['broad'] == 0, 'dist_2'] = -df_1894['dist_netw']

# =====================================
# DEFINE CONTROLS FOR PANEL C
# =====================================
# For panel C, controls are: distance to neighborhood centroid,
# distance to closest: square, bank, street vent, presumed plague pit
controls_1894 = [
    'dist_cent',      # distance to neighborhood centroid
    'dist_square',    # distance to square
    'dist_bank',      # distance to bank
    'dist_vent',      # distance to street vent
    'dist_pit_fake'   # distance to presumed plague pit
]

# =====================================
# DATA CLEANING
# =====================================
# Drop missing values - using rentals_94 in levels
df_clean = df_1894.dropna(subset=['rentals_94', 'dist_2', 'block'] + controls_1894)

# =====================================
# DETERMINE OPTIMAL BANDWIDTH
# =====================================
# You can calculate dynamically with rdrobust or use a fixed value
# Uncomment below to calculate dynamically:

# from rdrobust import rdrobust
# rd_result_c2 = rdrobust(
#     y=df_clean['log_rentals_1894'], 
#     x=df_clean['dist_2'], 
#     covs=df_clean[controls_1894], 
#     cluster=df_clean['block']
# )
# optimal_bw = rd_result_c2.bws.iloc[0, 0]

# Using default bandwidth (adjust based on actual rdrobust results)
optimal_bw = 0.2854  # Adjust this value based on actual rdrobust results for 1894 data

# =====================================
# REPLICATION FRAMEWORK - PANEL C COLUMN 2
# =====================================
# Filter to optimal bandwidth
df_bw = df_clean[np.abs(df_clean['dist_2']) <= optimal_bw].copy()
df_bw['treatment_rd'] = (df_bw['dist_2'] > 0).astype(int)

# Calculate triangular kernel weights (matching rdrobust default)
rd_weights = 1 - np.abs(df_bw['dist_2'].values) / optimal_bw

# Prepare variables - USE RENTALS_94 IN LEVELS
y_c2 = df_bw['rentals_94'].values  # 1894 rental values in LEVELS
X_c2 = sm.add_constant(df_bw[['treatment_rd', 'dist_2', 'dist_netw2'] + controls_1894], prepend=False)
cluster_c2 = df_bw['block'].values

# Metadata for Panel C Column 2
metadata_c2 = {
    'paper_id': '005',
    'table_id': '3',
    'panel_identifier': 'C_2',
    'model_type': 'log-linear',
    'comments': f'Table 3 Panel C Column 2: 1894 rentals, LLR with controls (Panel C control set); Bandwidth: {optimal_bw*100:.2f}m'
}

# Run replication
replicate(
    metadata=metadata_c2,
    y=y_c2,  # rentals_94 in LEVELS
    X=X_c2,
    interest='treatment_rd',
    elasticity=False,
    fe=None,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_c2}},
    weights=rd_weights
)