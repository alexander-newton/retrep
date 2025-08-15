import yaml
import os
import pandas as pd
import numpy as np
import sys

# Add path to replication module if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# =====================================
# LOAD DATA
# =====================================
filepath = os.path.join(INPUT_DATA_DIR, '006/final_data.dta')
df = pd.read_stata(filepath)

# Filter to main sample
df = df[df['sample_main'] == 1].copy()

# =====================================
# PREPARE VARIABLES
# =====================================

# Dependent variable: Speed of reactivation
y = df['time_to_reactivation'].values

# Treatment variable D (endogenous - share of time deactivated)
D = df['D'].values.reshape(-1, 1)

# Instrument T (random assignment)
T = df['T'].values.reshape(-1, 1)

# Control variables
control_vars = ['age', 'male', 'white', 'college', 'employed', 
                'household_income_coded', 'rep_party', 'dem_party']
controls = [c for c in control_vars if c in df.columns]

# Create X matrix: endogenous variable first, then controls, then constant
# The framework expects endogenous variables to be included in X
if controls:
    X_controls = df[controls].values
    X = np.column_stack([D, X_controls])
else:
    X = D

# Add constant (framework expects this)
X = np.column_stack([X, np.ones((len(df), 1))])

# =====================================
# METADATA
# =====================================
metadata = {
    'paper_id': 'facebook_deactivation',
    'table_id': 'A11',
    'panel_identifier': 'speed_reactivation',
    'model_type': 'log-linear', 
    'comments': 'Speed of reactivation (-ln(1+days)) - Table A11'
}
