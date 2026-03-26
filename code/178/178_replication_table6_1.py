import numpy as np
import os
import sys
import json
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from replication import replicate
from utils.json_utils import ReplicationJSONBuilder, load_config

# Load configuration
config = load_config()
raw_data_folder = config.get('rawdata', './rawdata')
intermediate_data_folder = config.get('intermediatedata', './intermediate_data')
final_data_folder = config.get('finaldata', './final_data')
output_folder = config.get('output', './output')

# Load data
df = pd.read_stata('/Users/ellenmunroe/Downloads/113312-V1/alcohol_data.dta')

# Convert categorical columns to numeric
for col in df.columns:
    if df[col].dtype.name == 'category':
        df[col] = df[col].cat.codes

# Drop Hawaii (state==15) and West Virginia (state==54)
df = df[(df['state'] != 15) & (df['state'] != 54)]

# Sort by state and year for panel operations
df = df.sort_values(['state', 'year']).reset_index(drop=True)

# Construct variables following alcohol_analysis.do
df['excise_price'] = 1 + df['beer_tax'] / 100
df['salestax_price'] = 1 + df['salestax'] / 100
df['beer_per_cap'] = df['c_beer'] / df['population']

# Take logs
df['ln_excise_price'] = np.log(df['excise_price'])
df['ln_salestax_price'] = np.log(df['salestax_price'])
df['ln_cons_beer'] = np.log(df['beer_per_cap'])
df['ln_population'] = np.log(df['population'])
df['ln_income'] = np.log(df['st_income'] / df['population'])
df['ln_unemp_rate'] = np.log(df['st_uemp_rate'])

# First differences (within each state)
for var in ['ln_excise_price', 'ln_salestax_price', 'ln_cons_beer', 'ln_population', 'ln_income', 'ln_unemp_rate']:
    df[f'd{var}'] = df.groupby('state')[var].diff()

# Drop missing from first-differencing
sample_vars = ['dln_cons_beer', 'dln_excise_price', 'dln_salestax_price', 'dln_population', 'dln_income', 'dln_unemp_rate', 'year']
df = df.dropna(subset=sample_vars)

# =====================================
# Table 6, Column 1: Baseline
# areg dln_cons_beer dln_excise_price dln_salestax_price dln_population, a(year)
# =====================================

metadata_col1 = {
    'paper_id': '178',
    'table_id': 'Table6',
    'panel_identifier': 'Col1',
    'model_type': 'log-log'
}

# Exponentiate the log difference to put DV back in levels
# exp(dln_cons_beer) = exp(ln(beer_t/beer_{t-1})) = beer_t/beer_{t-1}
y_col1 = np.exp(df['dln_cons_beer'].values).reshape(-1, 1)
X_col1 = df[['dln_excise_price', 'dln_salestax_price', 'dln_population']].values
fe_col1 = df[['year']].values

replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest='dln_excise_price',
    endog_x=None,
    z=None,
    fe=fe_col1,
    elasticity=True,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols=None,
    kwargs_ppml=None,
    fit_full_model=False,
    output=True,
    output_dir=output_folder
)
