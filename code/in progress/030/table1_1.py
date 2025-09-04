import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '030/Section4Data.xlsx')
df = pd.read_excel(filepath, sheet_name='Sheet1', skiprows=1)

# Filter for scheduled meetings only (excluding unscheduled FOMC statements)
# Column index 10 corresponds to 'unsched' column
df = df[df.iloc[:, 10] == 0].reset_index(drop=True)

# =====================================
# Table 1: Full Sample (Feb 2000 to May 2006)
# S&P 500 response to forward guidance
# =====================================

# S&P 500 column (index 33) contains percentage returns
sp500_returns = df.iloc[:, 33].values.astype(float)

# To get the correct signs, use negative of returns
y_full = np.exp(-sp500_returns/100.0) ** 100

# Define independent variables
x0 = df.iloc[:, 5].values.astype(float)  # Current fed funds surprise
x1 = df.iloc[:, 6].values.astype(float)
x2 = df.iloc[:, 7].values.astype(float)
x3 = df.iloc[:, 8].values.astype(float)
xP = np.mean([x1, x2, x3], axis=0)  # Forward guidance proxy

# Combine into X matrix
X_full = np.column_stack([x0, xP])
X_full = sm.add_constant(X_full, prepend=False)

# METADATA FOR FULL SAMPLE
metadata_full = {
    'paper_id': '030',
    'table_id': '1',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 1: S&P 500 response to Fed Funds and Forward Guidance surprises (Feb 2000 - May 2006)'
}

# RUN THE REPLICATION FOR FULL SAMPLE
replicate(
    metadata=metadata_full,
    y=y_full,
    X=X_full,
    interest=[0, 1],
    elasticity=False,
    fe=None,
    #kwargs_ols={'cov_type': 'HAC', 'cov_kwds': {'maxlags': 1}},

)