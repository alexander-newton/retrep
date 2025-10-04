import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# 1) Load config
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# 2) Load processed data
df = pd.read_stata(os.path.join(INPUT_DATA_DIR, '084/table5_baseline_prepared.dta')).sort_values('year')

# 3) Create required variables for Column (4)
df['incineq'] = df['top10incomeshare']           # no condition — always create

# 4) Build regression dataset
reg = df[['top10growth', 'housegrowth', 'stockgrowth', 'incineq', 'Y2W']].dropna()

# Estimator logs y internally → pass level ratio (>0)
y = np.exp(reg['top10growth'])

# Keep X as a named DataFrame so 'interest' works
X = reg[['housegrowth', 'stockgrowth', 'incineq', 'Y2W']].copy()
X = sm.add_constant(X, prepend=False)

metadata = {
    'paper_id': '084',
    'table_id': '5',
    'panel_identifier': '4',
    'model_type': 'log-linear',
    'comments': 'Table 5, Col (4): house/stock growth + income inequality + Y2W; y passed as ratio (estimator logs).'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=['housegrowth', 'stockgrowth'],
    elasticity=False,
    fe=None,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
