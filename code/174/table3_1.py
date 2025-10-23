import yaml
import os
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm

# Import your helper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Load config
# -------------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# -------------------------------
# Load data (Folder 174)
# -------------------------------
filepath = os.path.join(INPUT_DATA_DIR, '174/county_level_data.dta')
df = pd.read_stata(filepath)

# -------------------------------
# Sample: only 1950 and 2010 (as in Stata code)
# -------------------------------
df = df[df['year'].isin([1950, 2010])].copy()

# -------------------------------
# Prepare variables (use your exact names)
# Stata model: reghdfe lland_value_acre ee loo ee_innov, absorb(id year#state) cluster(id state_year)
# replicate() logs Y internally -> pass LEVELS of land_value_acre
# -------------------------------
required = [
    'year', 'id', 'state_year',
    'ee', 'loo', 'ee_innov',
    'lland_value_acre'
]
df_clean = df.dropna(subset=required).copy()

# Dependent variable in levels (since replicate logs internally)
y_levels = np.exp(df_clean['lland_value_acre'].astype(float))

# Build FE codes: county FE (id) and state×year FE (state_year)
df_clean['id_fe'] = pd.Categorical(df_clean['id']).codes
df_clean['stateyear_fe'] = pd.Categorical(df_clean['state_year']).codes

# X matrix: main regressors + FE placeholders (absorbed by replicate via fe indices)
x_vars = ['ee', 'ee_innov','loo', 'id_fe', 'stateyear_fe']
X = sm.add_constant(df_clean[x_vars].copy(), prepend=False)

# FE indices inside X
fe_idx = [X.columns.get_loc('id_fe'), X.columns.get_loc('stateyear_fe')]

# -------------------------------
# Metadata
# -------------------------------
metadata = {
    'paper_id': '174',
    'table_id': '3',
    'panel_identifier': '1',
    'model_type': 'log-linear',  # log DV (taken by replicate internally)
    'comments': 'Table 3, Column 1: log(land value per acre) on ee, loo, ee_innov with county FE (id) and state×year FE.'
}


replicate(
    metadata=metadata,
    y=y_levels,
    X=X,
    interest='ee_innov',
    fe=fe_idx,
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)


