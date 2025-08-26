import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '062/clean_IFLS.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 4 Column 6: 2SLS Estimates
# Dependent Variable: Log Bride Price Amount
# Instrument: School construction program (Duflo 2001)
# =====================================

# DATA PREPARATION
# Restrict to Duflo sample
df['birthyr'] = df['year'] - df['wife_age']
df = df[(df['birthyr'] >= 1950) & (df['birthyr'] <= 1972)].copy()

# Create intensity and age_1974
df['intensity'] = (df['totin'] / df['num_child']) * 1000
df['age_1974'] = 1974 - df['birthyr']
df.loc[df['age_1974'] >= 13, 'age_1974'] = 13

# Create instruments
for x in range(2, 13):
    df[f'int_{x}'] = np.where(df['age_1974'] == x, df['intensity'], 0)

# Filter sample
df = df[(df['bride_price'] == 1) & (df['ethnicity'] != 9)].copy()
print(f"After bride_price and ethnicity filter: {len(df)} observations")

df = df[df['dowry_value'] > 0].copy()
print(f"After removing zero dowry: {len(df)} observations")

# Create Duflo controls
birthyr_dummies = pd.get_dummies(df['birthyr'], prefix='birthyr')
duflo_control_cols = []
for col in birthyr_dummies.columns:
    for var in ['num_child', 'en71', 'wsppc']:
        new_col = f'{col}_{var}'
        df[new_col] = birthyr_dummies[col] * df[var]
        duflo_control_cols.append(new_col)


# Define variables
instruments = [f'int_{x}' for x in range(2, 13)]
controls = ['mar_age', 'mar_age2', 'mar_year', 'mar_year2']  # v107 will be FE only

# Check for missing values before dropna
print(f"\nMissing values check:")
for col in ['v107', 'clustervar', 'birthpl'] + controls:
    if col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing} missing")

# Keep all needed variables
keep_cols = ['dowry_value', 'elementary_plus', 'year', 'v107', 'clustervar', 'birthpl'] + instruments + controls + duflo_control_cols

df = df[keep_cols].dropna()
print(f"After dropna: {len(df)} observations")
df = df.reset_index(drop=True)

# Dependent variable
y_col6 = df['dowry_value']

# Save cluster variable for SE calculation
cluster_col6 = df['birthpl'].values

# Instruments
z_col6 = df[instruments]

# X matrix with controls and categorical FE variables (not dummies)
X_col6 = pd.concat([
    df[['elementary_plus']],
    df[controls],  # mar_age, mar_age2, mar_year, mar_year2
    df[duflo_control_cols],
    df[['year', 'v107', 'clustervar', 'birthpl']]  # Keep as categorical
], axis=1)

X_col6 = sm.add_constant(X_col6, prepend=False)

# Get indices of FE columns in X
fe_indices = []
for fe_var in ['year', 'v107', 'clustervar', 'birthpl']:
    fe_indices.append(X_col6.columns.get_loc(fe_var))

# METADATA
metadata_col6 = {
    'paper_id': '062',
    'table_id': '4',
    'panel_identifier': '6',
    'model_type': 'log-linear',
    'comments': 'Table 4 Column 6: 2SLS with full controls and FE.'
}
print(f"Observations before FE transformation: {len(y_col6)}")
# RUN REPLICATION
replicate(
    metadata=metadata_col6,
    y=y_col6,
    X=X_col6,
    interest='elementary_plus',
    endog_x=[0],  # Index of elementary_plus
    z=z_col6,
    fe=fe_indices,
    elasticity=False,
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col6}},
    output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True

)