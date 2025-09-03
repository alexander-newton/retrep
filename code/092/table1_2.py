import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# =========================
# Load configuration
# =========================
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# =========================
# Load data
# =========================
fp = os.path.join(INPUT_DATA_DIR, '092/eu_dataset.dta')
df = pd.read_stata(fp)

# =========================
# Create event study variables
# =========================
W = 10
tcol = 'event_time_yn'
shock = 'r_d_log_vat_rate'

# Create lag and lead variables
for i in range(0, W + 1):
    # lags: when event_time == -i
    df[f'vat_event_lag{i}'] = 0.0
    mask = (df[tcol] == -i)
    df.loc[mask, f'vat_event_lag{i}'] = df.loc[mask, shock]
    
    # leads: when event_time == +i  
    df[f'vat_event_lead{i}'] = 0.0
    mask = (df[tcol] == i)
    df.loc[mask, f'vat_event_lead{i}'] = df.loc[mask, shock]

# Drop lag0 and lag1 (as in Stata code)
df = df.drop(columns=['vat_event_lag0', 'vat_event_lag1'], errors='ignore')

# =========================
# COLUMN 2: VAT DECREASES ONLY
# =========================
# Filter for decreases (event_incr != 1 means exclude non-decreases)
df_decreases = df[df['event_incr'] != 1].copy()

# Define variables in Table 1 order (alternating pattern)
event_vars = [
    'vat_event_lead0',  # β₀
    'vat_event_lead1',  # β₊₁
    'vat_event_lag2',   # β₋₂
    'vat_event_lead2',  # β₊₂
    'vat_event_lag3',   # β₋₃
    'vat_event_lead3',  # β₊₃
    'vat_event_lag4',   # β₋₄
    'vat_event_lead4',  # β₊₄
]

# Add remaining leads (5-10)
for i in range(5, W + 1):
    event_vars.append(f'vat_event_lead{i}')

# Add remaining lags (5-10)
for i in range(5, W + 1):
    event_vars.append(f'vat_event_lag{i}')

controls = ['d_urate', 'd_inter', 'd_gdp_pc']

# Prepare data - keep only complete cases
vars_needed = ['d_log_price'] + event_vars + controls + ['time']
df_clean = df_decreases[vars_needed].dropna()

# Set up regression variables
y = np.exp(df_clean['d_log_price'].values)

# Include time in X matrix
X_vars = df_clean[event_vars + controls + ['time']]
X = sm.add_constant(X_vars, has_constant='add')

# Create metadata
metadata = {
    'paper_id': '092',
    'table_id': '1', 
    'panel_identifier': '2',
    'model_type': 'log-log'
}

# Run replication
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=0,  # Index of vat_event_lead0
    fe=['time'],  # Specify time as FE
    elasticity=True,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)