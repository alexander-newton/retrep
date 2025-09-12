import yaml
import os
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '110/master_SL.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 2 Panel A Column 3: All Conflict - Full Controls
# Dependent variable: ln_conflict_fatal (log of fatal conflict incidents)
# Main regressor: SL (slave trade variable)
# Controls: Country FE + Geographic controls + Historical controls
# Specification: reg ln_conflict_fatal SL i.ccode ln_shape_area ln_dist_kilometers meanalt SI abs_lat lon split10pc sc_mean v33 v30 patrilineal, r
# =====================================

# Define variable groups as per Stata code
geo_controls = ['ln_shape_area', 'ln_dist_kilometers', 'meanalt', 'SI', 'abs_lat', 'lon', 'split10pc', 'sc_mean']
hist_controls = ['v33', 'v30', 'patrilineal']

# Define all required variables for this regression
required_vars = ['ln_conflict_fatal', 'SL', 'ccode'] + geo_controls + hist_controls

# Drop missing values for required variables
df_clean = df.dropna(subset=required_vars)

print(f"Sample size after dropping missing values: {len(df_clean)}")

# Convert categorical ccode to numeric codes for FE algorithm
df_clean = df_clean.copy()
df_clean['ccode_numeric'] = df_clean['ccode'].cat.codes

# Dependent variable: ln_conflict_fatal (already logged)
# Convert back to levels since replicate function logs internally
y = np.exp(df_clean['ln_conflict_fatal'])

# Main treatment variable: SL (slave trade)
main_treatment = ['SL', 'v33']

# X matrix: treatment + geographic controls + historical controls + country FE (numeric)
X_vars = [main_treatment] + geo_controls + hist_controls + ['ccode_numeric']
X = df_clean[X_vars].copy()

# Add constant
X = sm.add_constant(X, prepend=False)

# Get the indices of FE columns in X (ccode_numeric is the country fixed effect)
fe_indices = [X.columns.get_loc('ccode_numeric')]

# Cluster variable (if clustering is desired - original Stata uses only robust SE)
cluster = df_clean['ccode_numeric']  # Could cluster by country

# METADATA
metadata = {
    'paper_id': '110',
    'table_id': '2',
    'panel_identifier': 'A_3',
    'model_type': 'log-linear',
    'comments': 'Table 2 Panel A Column 3: Fatal conflict incidents on slave trade (SL) with country FE, geographic controls, and historical controls'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='SL',  # Variable of interest
    fe=fe_indices,  # Country fixed effects
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
