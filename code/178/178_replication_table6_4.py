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

# Policy control variables
policy_vars = ['d_driving_age', 'd_bac_teen_02', 'd_lower_limit', 'd_bac_10', 'd_bac_08', 'd_admin_license_rev']

# Drop missing from first-differencing
sample_vars = ['dln_cons_beer', 'dln_excise_price', 'dln_salestax_price', 'dln_population',
               'dln_income', 'dln_unemp_rate', 'year', 'region'] + policy_vars
df = df.dropna(subset=sample_vars)

# =====================================
# Table 6, Column 4: Add Region Trends
# xi i.year
# areg dln_cons_beer dln_excise_price dln_salestax_price dln_population dln_income dln_unemp_rate d_* _I*, a(region)
# Note: year dummies are regressors, region is the absorbed FE
# =====================================

metadata_col4 = {
    'paper_id': '178',
    'table_id': 'Table6',
    'panel_identifier': 'Col4',
    'model_type': 'log-log'
}

# Exponentiate the log difference to put DV back in levels
y_col4 = pd.DataFrame(np.exp(df['dln_cons_beer'].values), columns=['dln_cons_beer'])

# Create year dummies (xi i.year in Stata) — these are regressors, not FE
year_dummies = pd.get_dummies(df['year'].astype(int), prefix='yr', drop_first=True).astype(float).reset_index(drop=True)

# X matrix: interest variable first, then other regressors, policy controls, year dummies, then FE (region)
x_regressors = ['dln_excise_price', 'dln_salestax_price', 'dln_population', 'dln_income', 'dln_unemp_rate'] + policy_vars
X_col4 = pd.concat([
    df[x_regressors].reset_index(drop=True),
    year_dummies,
    df[['region']].reset_index(drop=True)
], axis=1)

# Save to parquet
out_dir = os.path.join(output_folder, '178', 'Table6_Col4')
os.makedirs(out_dir, exist_ok=True)

y_col4.to_parquet(os.path.join(out_dir, 'y.parquet'), index=False)
X_col4.to_parquet(os.path.join(out_dir, 'X.parquet'), index=False)

# Save metadata
with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata_col4, f, indent=2)

# For replicate: everything except region is X, region is FE
year_dummy_cols = list(year_dummies.columns)
replicate(
    metadata=metadata_col4,
    y=y_col4.values,
    X=X_col4[x_regressors + year_dummy_cols].values,
    interest='dln_excise_price',
    endog_x=None,
    z=None,
    fe=X_col4[['region']].values,
    elasticity=True,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols=None,
    kwargs_ppml=None,
    fit_full_model=False,
    output=True,
    output_dir=output_folder
)
