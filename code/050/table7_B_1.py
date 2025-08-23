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
filepath = os.path.join(INPUT_DATA_DIR, '050/villageData.dta')
df = pd.read_stata(filepath)

# Control variables
control_vars = ['above_400', 'bigholdings']

# Baseline controls
baseline_controls = [
    'runvar_400_1', 'runvar_400_2', 'runvar_400_3',
    'runvar_400_1_X_above', 'runvar_400_2_X_above', 'runvar_400_3_X_above',
    'runvar_400_1_bigholdings', 'runvar_400_2_bigholdings', 'runvar_400_3_bigholdings',
    'runvar_400_1_X_triple', 'runvar_400_2_X_triple', 'runvar_400_3_X_triple'
]

# Combine controls (without FE)
all_controls = control_vars + baseline_controls

# Encode regency63 for clustering
df['regency63n'] = pd.Categorical(df['regency63']).codes

# Create AgGDPcapita in levels from the logged variable
df['AgGDPcapita'] = np.exp(df['lnAgGDPcapita']) # levels were not included in the dataset.

# Table 7 Panel B Column 1: OLS
# Using AgGDPcapita in levels
regression_vars = ['AgGDPcapita', 'lwaqfland_ha'] + all_controls + ['regency63n', 'isleID']
df_clean = df[regression_vars].dropna()

y = df_clean['AgGDPcapita'].values  # Using levels
X = df_clean[['lwaqfland_ha'] + all_controls + ['isleID']]
X = sm.add_constant(X)
cluster = df_clean['regency63n']

metadata = {
    'paper_id': '050',
    'table_id': '7',
    'panel_identifier': 'B_1',
    'model_type': 'log-linear',
    'comments': 'Table 7 Panel B Column 1: OLS - Log Agricultural GDP per capita on waqf land'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,  # Pass as DataFrame, not .values
    interest='lwaqfland_ha',  # Pass column name, not index
    elasticity=False,
    fe=['isleID'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)