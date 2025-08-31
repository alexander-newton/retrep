"""
Table 4: Beta bond returns on stock returns - Period 2
"""

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

# LOAD DATA from XLSX file
filepath = os.path.join(INPUT_DATA_DIR, '080/DataTable_2017.xlsx')
df = pd.read_excel(filepath, sheet_name='Data', engine='openpyxl')

# Create quarter index
df['q'] = pd.to_datetime(df['qdate']).dt.year * 4 + pd.to_datetime(df['qdate']).dt.quarter - 7840

# Calculate log returns as in Stata
df['xr_bond_log'] = (-19 * df['yield5'] + 20 * df['yield5'].shift(1) - df['tbill'].shift(1)) / 4
df['xr_equity'] = (df['vwret'] - df['tbill'].shift(1)) / 4

# Convert bond log returns to levels for y (will be logged internally by replicate)
df['xr_bond'] = np.exp(df['xr_bond_log'])

# Period 2: q>=165 & q<208
df_period2 = df[(df['q'] >= 165) & (df['q'] < 208)].dropna(subset=['xr_bond', 'xr_equity'])

metadata = {
    'paper_id': '080',
    'table_id': '4',
    'panel_identifier': 'beta_bond_p2',
    'model_type': 'log-linear',  # log(y) on X (where X is already log returns)
    'comments': 'Beta bond on stock, Period 2 (01:Q2-11:Q4). Bond returns in levels (logged internally), equity returns already in logs.'
}

replicate(
    metadata=metadata,
    y=df_period2['xr_bond'],  # In levels (will be logged internally)
    X=sm.add_constant(df_period2[['xr_equity']], prepend=False),  # Already in log returns
    interest='xr_equity',
    elasticity=False,  # False since X is not being logged
    output=True, output_dir=OUTPUT_DIR, replicated=True
)