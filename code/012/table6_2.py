import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '012/diagnoza_replicate.dta')
df = pd.read_stata(filepath)

# Convert categorical columns to numeric
for col in df.columns:
    if df[col].dtype.name == 'category':
        df[col] = df[col].cat.codes

# Define base controls first
base_controls = [
    'female',
    'age_birth_pre1930', 'age_sq_birth_pre1930',
    'age_birth_1930s', 'age_sq_birth_1930s',
    'age_birth_1940s', 'age_sq_birth_1940s',
    'age_birth_1950s', 'age_sq_birth_1950s',
    'age_birth_1960s', 'age_sq_birth_1960s',
    'age_birth_1970s', 'age_sq_birth_1970s',
    'age_birth_1980s', 'age_sq_birth_1980s',
    'age_birth_1990s', 'age_sq_birth_1990s',
    'rural'
]

# Create the exact sample from the first regression (before standardization)
# This matches: areg log_hhincome anybody_from_kresy edu_years $base_controls pow_FE_indicator if pensioner_status==0
sample_vars = ['log_hhincome', 'anybody_from_kresy', 'edu_years'] + base_controls + ['pow_FE_indicator', 'powiaty_code', 'hh']
df = df[df['pensioner_status'] == 0].copy()
df = df.dropna(subset=sample_vars)

# Now standardize education years within this exact sample (matching Stata's e(sample))
df['edu_years_std'] = (df['edu_years'] - df['edu_years'].mean()) / df['edu_years'].std()
df['i_kresy_edu_std'] = df['anybody_from_kresy'] * df['edu_years_std']

# No additional dropna needed - already filtered above
# all_vars already checked in sample_vars

# Dependent variable
y_col2 = np.exp(df['log_hhincome'])

# X matrix: main variables first, then controls  
X_vars = ['anybody_from_kresy', 'edu_years_std', 'i_kresy_edu_std'] + base_controls + ['pow_FE_indicator', 'powiaty_code']
X_col2 = sm.add_constant(df[X_vars], prepend=False)

# Cluster variable
cluster_col2 = df['hh']

# METADATA
metadata_col2 = {
    'paper_id': '012',
    'table_id': '6',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comments': 'Table 6 Column 2: Differential returns to schooling'
}

# RUN REPLICATION
replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest='i_kresy_edu_std',
    elasticity=False,
    fe=['powiaty_code'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
    
   # output=True, output_dir=OUTPUT_DIR, replicated=True
)