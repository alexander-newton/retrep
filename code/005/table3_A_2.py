import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from rdrobust import rdrobust

# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from replication import replicate

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
df_clean = df.dropna(subset=['log_rentals_1853', 'dist_2', 'block'] + controls)

# =====================================
# RUN RDROBUST WITH CONTROLS
# =====================================
rd_result = rdrobust(
   y=df_clean['log_rentals_1853'], 
   x=df_clean['dist_2'],
   covs=df_clean[controls],
   cluster=df_clean['block'],
   all=True
)

# =====================================
# EXTRACT AND DISPLAY RESULTS
# =====================================
# Extract estimates
coef = rd_result.coef.iloc[0, 0]
se = rd_result.se.iloc[0, 0]
pval = rd_result.pv.iloc[0, 0]
ci_lower = rd_result.ci.iloc[0, 0]
ci_upper = rd_result.ci.iloc[0, 1]
obs = rd_result.N_h[0] + rd_result.N_h[1]
bw = rd_result.bws.iloc[0, 0]
bw_meters = bw * 100

# Calculate mean outside BSP within bandwidth
mean_outside = df.loc[
   (df['broad'] == 0) & (df['dist_netw'] <= bw), 
   'rentals_53'
].mean()

# Display results
print(f"\nInside BSP area:     {coef:.3f}")
print(f"Standard error:     ({se:.3f})")
print(f"P-value:            {pval:.3f}")
print(f"95% CI:             [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"\nObservations:       {int(obs)}")
print(f"Mean outside BSP:   {mean_outside:.2f}")
print(f"Bandwidth (meters): {bw_meters:.2f}")

# Compare with paper
print("\n" + "-"*40)
print("Expected from paper:")
print("Coefficient: 0.035 (0.078)")
print("Observations: 469")
print("Bandwidth: 27.25 meters")
print("="*60)

# =====================================
# REPLICATION FRAMEWORK FORMAT
# =====================================
# # Filter to optimal bandwidth
optimal_bw = rd_result.bws.iloc[0, 0]
df_bw = df_clean[np.abs(df_clean['dist_2']) <= optimal_bw].copy()
df_bw['treatment_rd'] = (df_bw['dist_2'] > 0).astype(int)

# Prepare variables
y_col2 = df_bw['log_rentals_1853']
X_col2 = sm.add_constant(df_bw[['treatment_rd', 'dist_2', 'dist_netw2'] + controls], prepend=False)
cluster_col2 = df_bw['block']

# Metadata
metadata_col2 = {
    'paper_id': '005',
    'table_id': '3',
    'panel_identifier': 'A_2',
    'model_type': 'log-linear',
    'comments': f'Table 3 Panel A Column 2: LLR with controls; Optimal bandwidth: {optimal_bw*100:.2f}m'
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
)