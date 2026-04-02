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
output_folder = config.get('output', './output')

# =====================================
# Bazzi & Blattman (2014) - Table 7, Column 4
# Log interval regression of battle deaths on commodity price shocks (no LDV)
# intreg ltotbdead_int_lo ltotbdead_int_hi pshock_npi_p pshock_npi_p1 pshock_npi_p2
#        duration first_year mena ssa asia americas Iy*
#        if (state_exist==1 | state_exist_cow==1), cl(cow_id)
# =====================================

# Load data
df = pd.read_stata('/Users/ellenmunroe/Downloads/114306-V1/Data/BAZZI_BLATTMAN_FINAL.dta')

# Convert categorical columns to numeric
for col in df.columns:
    if df[col].dtype.name == 'category':
        df[col] = df[col].cat.codes

# Panel setup
df = df.sort_values(['cow_id', 'year']).reset_index(drop=True)

# Construct mena = mideast + nafrica
df['mena'] = df['mideast'] + df['nafrica']

# Impute missing battle deaths as average of high and low
df['impute'] = ((df['any_prio'] == 1) & df['totbdeadbes'].isna()).astype(float)
df['totbdeadbes_impute'] = df['totbdeadbes'].copy()
mask_impute = (df['any_prio'] == 1) & df['totbdeadbes_impute'].isna()
df.loc[mask_impute, 'totbdeadbes_impute'] = (df.loc[mask_impute, 'totbdeadhig'] + df.loc[mask_impute, 'totbdeadlow']) / 2

# Create interval measure: use best estimate where available, else hi/lo
df['totbdead_int_hi'] = df['totbdeadbes'].copy()
mask_hi = df['totbdead_int_hi'].isna() & (df['any_prio'] == 1)
df.loc[mask_hi, 'totbdead_int_hi'] = df.loc[mask_hi, 'totbdeadhig']

df['totbdead_int_lo'] = df['totbdeadbes'].copy()
mask_lo = df['totbdead_int_lo'].isna() & (df['any_prio'] == 1)
df.loc[mask_lo, 'totbdead_int_lo'] = df.loc[mask_lo, 'totbdeadlow']

# Log battle deaths
df['ltotbdead_int_hi'] = np.log(df['totbdead_int_hi'])
df['ltotbdead_int_lo'] = np.log(df['totbdead_int_lo'])

# First year and duration
df['first_year'] = np.where(
    (df['any_prio'] == 1),
    np.where(df['totbdeadbes_impute'].notna() & df['end_any_prio'].isna(), 1, 0),
    np.nan
)

df['duration'] = df['first_year'].copy()
for _, grp in df.groupby('cow_id'):
    idx = grp.index
    for j in range(1, len(idx)):
        if df.loc[idx[j], 'first_year'] != 1 and df.loc[idx[j], 'any_prio'] == 1:
            prev = df.loc[idx[j-1], 'duration']
            if not np.isnan(prev):
                df.loc[idx[j], 'duration'] = prev + 1

# Year dummies
year_dummies = pd.get_dummies(df['year'].astype(int), prefix='Iy', drop_first=True).astype(float)
year_cols = list(year_dummies.columns)
df = pd.concat([df, year_dummies], axis=1)

# Sample restriction: state_exist==1 | state_exist_cow==1
df = df[(df['state_exist'] == 1) | (df['state_exist_cow'] == 1)].copy()

# For column 4: DV is log battle deaths (use midpoint of interval as point estimate)
# y in levels = exp(midpoint of log interval) = geometric mean of interval bounds
df['ltotbdead_mid'] = (df['ltotbdead_int_lo'] + df['ltotbdead_int_hi']) / 2
df['totbdead_mid'] = np.exp(df['ltotbdead_mid'])

# Variables for regression
x_vars = ['pshock_npi_p', 'pshock_npi_p1', 'pshock_npi_p2',
           'duration', 'first_year', 'mena', 'ssa', 'asia', 'americas'] + year_cols

sample_vars = ['totbdead_mid', 'ltotbdead_mid'] + x_vars
reg_df = df[sample_vars].dropna().reset_index(drop=True)

# =====================================
# Table 7, Column 4
# =====================================

metadata = {
    'paper_id': '180',
    'table_id': 'Table7',
    'panel_identifier': 'Col4',
    'model_type': 'log-linear'
}

# y in levels (exponentiated log battle deaths)
y = pd.DataFrame(reg_df['totbdead_mid'].values, columns=['totbdead'])

# X: interest variable first (pshock_npi_p), then controls, year FE
X = reg_df[x_vars].reset_index(drop=True)

# Save to parquet
out_dir = os.path.join(output_folder, '180', 'Table7_Col4')
os.makedirs(out_dir, exist_ok=True)

y.to_parquet(os.path.join(out_dir, 'y.parquet'), index=False)
X.to_parquet(os.path.join(out_dir, 'X.parquet'), index=False)

with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# replicate() call
replicate(
    metadata=metadata,
    y=y.values,
    X=X[['pshock_npi_p', 'pshock_npi_p1', 'pshock_npi_p2',
         'duration', 'first_year', 'mena', 'ssa', 'asia', 'americas']].values,
    interest='pshock_npi_p',
    endog_x=None,
    z=None,
    fe=X[year_cols].values,
    elasticity=True,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols=None,
    kwargs_ppml=None,
    fit_full_model=False,
    output=True,
    output_dir=output_folder
)
