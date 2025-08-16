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
# LOAD DATA
# =====================================
filepath = os.path.join(INPUT_DATA_DIR, '005/Merged_1853_1864_data.dta')
df = pd.read_stata(filepath)

# =====================================
# DATA PREPARATION
# =====================================
# Scale distance and create RD running variable
df['dist_netw'] = df['dist_netw'] / 100
df['dist_netw2'] = df['dist_netw'] ** 2

# Create running variable (negative outside BSP, positive inside)
df['dist_2'] = df['dist_netw'].copy()
df.loc[df['broad'] == 0, 'dist_2'] = -df['dist_netw']

# Define control variables
controls = ['dist_cent', 'dist_square', 'dist_fire', 'dist_thea', 'dist_police', 
          'dist_urinal', 'dist_pub', 'dist_church', 'dist_bank', 'no_sewer', 
          'old_sewer', 'dist_vent', 'dist_pump', 'dist_pit_fake']

# Drop missing values
df_clean = df.dropna(subset=['rentals_53', 'dist_2', 'block'] + controls)

# =====================================
# DETERMINE OPTIMAL BANDWIDTH
# =====================================
# Using default bandwidth of 0.2725 (27.25 meters) as in the paper
# You can uncomment the rdrobust code below if you want to calculate it dynamically

# from rdrobust import rdrobust
# rd_result = rdrobust(
#    y=df_clean['log_rentals_1853'],
#    x=df_clean['dist_2'],
#    covs=df_clean[controls],
#    cluster=df_clean['block'],
#    all=True
# )
# optimal_bw = rd_result.bws.iloc[0, 0]

optimal_bw = 0.2725  # 27.25 meters in scaled units

# =====================================
# REPLICATION FRAMEWORK
# =====================================
# Filter to optimal bandwidth
df_bw = df_clean[np.abs(df_clean['dist_2']) <= optimal_bw].copy()
df_bw['treatment_rd'] = (df_bw['dist_2'] > 0).astype(int)

# Calculate triangular kernel weights (matching rdrobust default)
rd_weights = 1 - np.abs(df_bw['dist_2'].values) / optimal_bw

# Prepare variables
y_col2 = df_bw['rentals_53'].values  # Using rental values in LEVELS
X_col2 = sm.add_constant(df_bw[['treatment_rd', 'dist_2', 'dist_netw2'] + controls], prepend=False)
cluster_col2 = df_bw['block'].values

# Metadata
metadata_col2 = {
    'paper_id': '005',
    'table_id': '3',
    'panel_identifier': 'A_2',
    'model_type': 'log-linear',
    'comments': f'Table 3 Panel A Column 2: LLR with controls; Bandwidth: {optimal_bw*100:.2f}m'
}

# Run replication
replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest='treatment_rd',
    elasticity=False,
    fe=None,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
    weights=rd_weights,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)