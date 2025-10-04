import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ---------- Load processed data ----------
df = pd.read_stata(os.path.join(INPUT_DATA_DIR, '084/table5_baseline_prepared.dta')).sort_values('year')

# ---------- Baseline regression vars ----------
reg = df[['top10growth', 'housegrowth', 'stockgrowth']].dropna()

# y must stay as LOG growth (already log in the Stata-prepared file)
y = np.exp(reg['top10growth'])                                  # pandas Series

# Make X a DataFrame so columns are named and recognized
X = reg[['housegrowth', 'stockgrowth']].copy()          # DataFrame with names
X = sm.add_constant(X, prepend=False)                   # adds 'const' as the last column (still a DataFrame)

metadata = {
    'paper_id': '084',
    'table_id': '5',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 5, Column (1): log Δ top10 wealth share ~ log Δ houseprice + log Δ stockprice'
}

replicate(
    metadata=metadata,
    y=y,                               # Series with name preserved
    X=X,                               # DataFrame with named cols
    interest=['housegrowth','stockgrowth'],  # now recognized
    elasticity=False,
    fe=None,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
