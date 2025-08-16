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


# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '018/PAC_charity.dta')
df = pd.read_stata(filepath)

# Convert categorical columns to numeric
for col in df.columns:
    if df[col].dtype.name == 'category':
        df[col] = df[col].cat.codes

# Define variables needed
sample_vars = ['PACamount', 'lnrep_issue_state_cd', 
               'EIN_state_cd_id', 'state_cd_congress_id', 'EIN_congress_id']

# Drop missing values
df = df.dropna(subset=sample_vars)

# Dependent variable - use 1 + PACamount
y_col7 = 1 + df['PACamount']

# X matrix
X_vars = ['lnrep_issue_state_cd', 'EIN_state_cd_id', 'state_cd_congress_id', 'EIN_congress_id']
X_col7 = sm.add_constant(df[X_vars], prepend=False)

# Cluster variable
cluster_col7 = df['EIN_state_cd_id']

# METADATA
metadata_col7 = {
    'paper_id': '018',
    'table_id': '3',
    'panel_identifier': '7',
    'model_type': 'log-log'
}

# RUN REPLICATION
replicate(
    metadata=metadata_col7,
    y=y_col7,
    X=X_col7,
    interest='lnrep_issue_state_cd',
    elasticity=True,
    fe=['EIN_state_cd_id', 'state_cd_congress_id', 'EIN_congress_id'],
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col7}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)