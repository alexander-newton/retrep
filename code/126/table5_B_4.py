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

# Load city dataset (as per correct Stata script)
filepath = os.path.join(INPUT_DATA_DIR, '126/city_dataset.dta')
df = pd.read_stata(filepath)

# Define variables
imm_var = 'recentEU_predpop'
iv_var = 'ZEUMSA_predpop'

# Clean data
required_vars = [imm_var, iv_var, 'l_estab_size', 'city', 'metarea', 'stateyear']
df = df.dropna(subset=required_vars)

# Create levels from logged dependent variable (since replicate function logs internally)
y = np.exp(df['l_estab_size'])

# Convert FE variables to numeric codes for replicate function
df['city_code'] = pd.Categorical(df['city']).codes
df['stateyear_code'] = pd.Categorical(df['stateyear']).codes

# X matrix: endogenous variable + FE codes + constant
X = df[[imm_var, 'city_code', 'stateyear_code']].copy()
X = sm.add_constant(X, prepend=False)

# z matrix: instrument + SAME FE codes + constant  
z = df[[iv_var, 'city_code', 'stateyear_code']].copy()
z = sm.add_constant(z, prepend=False)

# Fixed effects variable names for replicate function
fe_vars = ['city_code', 'stateyear_code']

# Cluster variable (ensure it's numeric)
cluster = pd.Categorical(df['metarea']).codes

# Metadata
metadata = {
    'paper_id': '126',
    'table_id': '5',
    'panel_identifier': 'B_4',
    'model_type': 'log-linear',
    'comments': 'Table 5 Panel B Column 4: 2SLS regression of l_estab_size on recentEU_predpop with city + stateyear FE, clustered SE at MSA level'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=imm_var,
    endog_x=[imm_var],
    z=z,
    elasticity=False,
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
