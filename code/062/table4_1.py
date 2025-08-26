'''
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
# Table 4 Column 1: Determinants of Bride Price Payment Amounts
# Dependent Variable: Log Bride Price Amount
# Sample: All observations with valid ethnicity (excluding group 9)
# Baseline covariates: ethnicity FE, year married and its square
# =====================================

# DATA PREPARATION
# Following Stata code: reg log_dowry elementary_plus junior college i.ethnicity i.year `marpoly' if ethnicity!=9
df = df[df['ethnicity'] != 9].copy()

# Create marriage year squared if not present
if 'mar_year2' not in df.columns:
    df['mar_year2'] = df['mar_year'] ** 2

# Additional filter from Stata code: drop if dowry_gdp is missing
df = df[df['dowry_gdp'].notna()].copy()    


# Filter to observations with non-missing dowry_value and all required variables
keep_cols = ['dowry_value', 'elementary_plus', 'junior_plus', 'college', 
             'ethnicity', 'year', 'mar_year', 'mar_year2']
df = df[keep_cols].dropna()

# Remove zeros from dowry_value since we'll take log
df = df[df['dowry_value'] > 0].copy()

# Reset index
df = df.reset_index(drop=True)

# Dependent variable (in levels)
y_col1 = df['dowry_value']

# Main treatment variables
main_treatments = ['elementary_plus', 'junior_plus', 'college']

# Controls
control_vars = ['mar_year', 'mar_year2']

# Create dummy variables for fixed effects manually
ethnicity_dummies = pd.get_dummies(df['ethnicity'], prefix='ethnicity', drop_first=True)
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True)

# Get FE column names
fe_cols = list(ethnicity_dummies.columns) + list(year_dummies.columns)

# Create X matrix with dummies already included
X_col1 = pd.concat([
    df[main_treatments],
    df[control_vars],
    ethnicity_dummies,
    year_dummies
], axis=1)

X_col1 = sm.add_constant(X_col1, prepend=False)

# Get FE indices (positions of dummy variables in X matrix)
fe_indices = []
for col in fe_cols:
    fe_indices.append(X_col1.columns.get_loc(col))

# METADATA FOR THIS RESULT
metadata_col1 = {
    'paper_id': '062',
    'table_id': '4',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 4 Column 1: IFLS Hedonic regression. DV: Log bride price amount. Sample: ethnicity!=9, positive dowry values. Baseline covariates: ethnicity FE, year FE, marriage year polynomials. Robust SE.'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest='elementary_plus',
    fe=fe_indices,
    elasticity=False,

)
'''
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
# Table 4 Column 1: Determinants of Bride Price Payment Amounts
# Dependent Variable: Log Bride Price Amount
# Sample: All observations with valid ethnicity (excluding group 9)
# Baseline covariates: ethnicity FE, year married and its square
# =====================================

# DATA PREPARATION
df = df[df['eth_group'] != 9].copy()

# Create marriage year squared if not present
if 'mar_year2' not in df.columns:
    df['mar_year2'] = df['mar_year'] ** 2

# Additional filter from Stata code
df = df[df['dowry_gdp'].notna()].copy()

# Filter to observations with all required variables
keep_cols = ['dowry_value', 'elementary_plus', 'junior_plus', 'college', 
             'eth_group', 'year', 'mar_year', 'mar_year2']
df = df[keep_cols].dropna()

# Remove zeros from dowry_value since we'll take log
df = df[df['dowry_value'] > 0].copy()
df = df.reset_index(drop=True)

# Dependent variable
y_col1 = df['dowry_value']

# Main treatment variables
main_treatments = ['elementary_plus', 'junior_plus', 'college']

# Controls
control_vars = ['mar_year', 'mar_year2']

# X matrix with controls and categorical FE variables (NOT dummies)
X_col1 = pd.concat([
    df[main_treatments],
    df[control_vars],
    df[['eth_group', 'year']]  # Keep as categorical
], axis=1)

X_col1 = sm.add_constant(X_col1, prepend=False)

# Get indices of FE columns
fe_indices = [X_col1.columns.get_loc('eth_group'), X_col1.columns.get_loc('year')]

# METADATA
metadata_col1 = {
    'paper_id': '062',
    'table_id': '4',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 4 Column 1: IFLS Hedonic regression. Baseline covariates: ethnicity FE, year FE, marriage year polynomials. Robust SE.'
}

print(f"Observations before FE transformation: {len(y_col1)}")

# RUN REPLICATION
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest='elementary_plus',
    fe=fe_indices,
    elasticity=False,
)