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
filepath = os.path.join(INPUT_DATA_DIR, '125/Main_Tables.dta')
df = pd.read_stata(filepath)

controls_2sls = ['instind_avg', 'instgdp_avg', 'lat', 'lon', 'lat_long', 'lat2_long', 
                'lat_long2', 'lat3', 'long3', 'lat2', 'long2', 'lcohort']

endog_var = 'foreign_avg'
instrument_var = 'instmig_avg'
outcome = 'inc'

required_vars = [outcome, endog_var, instrument_var] + controls_2sls
df_clean = df.dropna(subset=[v for v in required_vars if v in df.columns])

state_vars = [col for col in df_clean.columns if col.startswith('stid') and len(col) <= 6]

y = np.exp(df_clean[outcome].values)

X_vars = [endog_var] + controls_2sls + state_vars
X = df_clean[X_vars].copy()
X = sm.add_constant(X, prepend=False)

z_vars = [instrument_var] + controls_2sls + state_vars
z = df_clean[z_vars].copy() 
z = sm.add_constant(z, prepend=False)

state_start_idx = X.columns.get_loc(state_vars[0])
state_end_idx = X.columns.get_loc(state_vars[-1])
fe_indices = list(range(state_start_idx, state_end_idx + 1))

metadata = {
    'paper_id': '125',
    'table_id': '4',
    'panel_identifier': 'C_1',
    'model_type': 'log-linear',
    'comments': 'Table 4 Panel C Column 1: 2SLS regression of income on foreign_avg, instrumented with instmig_avg'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=[endog_var],
    endog_x=[endog_var],
    z=z,
    fe=fe_indices,
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
