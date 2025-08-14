import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from rdrobust import rdrobust

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '005/Merged_1853_1864_data.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 2 Column 2: LLR with Controls
# Sample: Properties within optimal bandwidth
# =====================================

# Data preparation
df['temp'] = df['dist_netw'] / 100
df.loc[df['broad'] == 0, 'temp'] = -df.loc[df['broad'] == 0, 'dist_netw'] / 100

df['dist_netw'] = df['dist_netw'] / 100
df['dist_netw2'] = df['dist_netw'] ** 2
df['dist_netw3'] = df['dist_netw'] ** 3

df['dist_2'] = df['dist_netw']
df.loc[df['broad'] == 0, 'dist_2'] = -df.loc[df['broad'] == 0, 'dist_netw']

# Define controls
controls = ['dist_cent', 'dist_square', 'dist_fire', 'dist_thea', 'dist_police', 
           'dist_urinal', 'dist_pub', 'dist_church', 'dist_bank', 'no_sewer', 
           'old_sewer', 'dist_vent', 'dist_pump', 'dist_pit_fake']

# Drop missing values
df_clean = df.dropna(subset=['log_rentals_1853', 'dist_2', 'block'] + controls)

# Run rdrobust to get optimal bandwidth and coefficients
rd_result = rdrobust(y=df_clean['log_rentals_1853'], 
                     x=df_clean['dist_2'],
                     covs=df_clean[controls],
                     cluster=df_clean['block'],
                     all=True)

# Get optimal bandwidth
optimal_bw = rd_result.bws.iloc[0, 0]

# Filter data to optimal bandwidth for replication format
df_bw = df_clean[np.abs(df_clean['dist_2']) <= optimal_bw].copy()

# Create indicator for treatment (inside BSP area)
# In RD context, treatment is being on the positive side of the cutoff
df_bw['treatment_rd'] = (df_bw['dist_2'] > 0).astype(int)

# Dependent variable
y_col2 = df_bw['log_rentals_1853']

# Main treatment variable (RD treatment effect at cutoff)
main_treatment = 'treatment_rd'

# Running variable and controls
running_vars = ['dist_2', 'dist_netw2']
X_col2 = sm.add_constant(df_bw[[main_treatment] + running_vars + controls], prepend=False)

# Cluster variable
cluster_col2 = df_bw['block']

# METADATA FOR THIS RESULT
metadata_col2 = {
    'paper_id': 'cholera_2024',
    'table_id': '2',
    'panel_identifier': 'col2_llr_controls',
    'model_type': 'log-linear',
    'comments': f'Table 2 Column 2: LLR with controls; Log rentals 1853 RD analysis at BSP boundary; Optimal bandwidth: {optimal_bw*100:.2f} meters; Controls included: {", ".join(controls)}'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest=main_treatment,
    elasticity=False,
    fe=None,  # No fixed effects in this specification
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
  
)

