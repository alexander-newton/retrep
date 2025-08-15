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

# Load data
df = pd.read_stata(os.path.join(INPUT_DATA_DIR, '008/pmgsy_working_aer_mainsample.dta'))

# Filter main sample and remove NaNs
df = df[df['mainsample'] == 1].copy()

# Sector mapping
sectors = {
    'Total': 'ec13_emp_all_ln',
    'Livestock': 'ec13_emp_act1_ln',
    'Manufacturing': 'ec13_emp_act2_ln',
    'Education': 'ec13_emp_act3_ln',
    'Retail': 'ec13_emp_act6_ln',
    'Forestry': 'ec13_emp_act12_ln'
}

# Controls
controls = ['left', 'right', 'scst_share', 'bus', 'comm']

# Run each sector
for sector_name, y_var in sectors.items():
    # Skip if outcome variable doesn't exist
    if y_var not in df.columns:
        continue
    
    # Remove rows with missing values for this regression
    df_clean = df[[y_var, 'r2012', 't'] + controls + ['vhg_dist_id']].dropna()
    
    # Prepare variables
    y = df_clean[y_var].values
    
    # Build X matrix: controls + r2012 + district FEs + constant
    X_controls = df_clean[controls].values
    X_endog = df_clean[['r2012']].values
    district_fe = pd.get_dummies(df_clean['vhg_dist_id'], prefix='dist', drop_first=True).values
    
    # Combine all X variables
    X = np.column_stack([X_controls, X_endog, district_fe])
    X = sm.add_constant(X)
    
    # Instrument
    z = df_clean[['t']].values
    
    # Get r2012 index (after controls, before FEs)
    endog_idx = len(controls)
    
    # Metadata
    metadata = {
        'paper_id': '008',
        'table_id': '6',
        'panel_identifier': 'A_1',
        'model_type': 'log-linear',
        'comments': 'Impact of New Road on Firms: table 6 panel A column 1'
    }
    
    # Run replication
    replicate(
        metadata=metadata,
        y=y,
        X=X,
        interest=endog_idx,
        endog_x=[endog_idx],
        z=z,
        elasticity=False,
        replicated=False,
        #kwargs_ols={'cov_type': 'HC0'},
    )