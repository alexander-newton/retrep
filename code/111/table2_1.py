import yaml
import os
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '111/HetMarkupsGrowthMisallocation_FirmlevelData.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 2 Column 1: Basic markup lifecycle profile
# Stata: areg ln_sl_corr cohortage i.year if Sample == 1, absorb(mainproduct) robust
# Sample restrictions: Sample == 1, years 1991-2000 (as per table note: "data from 1991 to 2000")
# Note: The Stata snippet provided may be incomplete - using table note specification
# =====================================

# Apply sample restrictions based on table note
df_sample = df[df['Sample'] == 1].copy()
df_sample = df_sample[(df_sample['year'] >= 1991) & (df_sample['year'] <= 2000)].copy()

print(f"Sample size after restrictions: {len(df_sample)}")
print(f"Years included: {sorted(df_sample['year'].unique())}")

# Drop missing values for required variables
required_vars = ['ln_sl_corr', 'cohortage', 'year', 'mainproduct']
df_clean = df_sample.dropna(subset=required_vars)

print(f"Sample size after dropping missing values: {len(df_clean)}")

# Convert categorical variables to numeric for FE
df_clean = df_clean.copy()
df_clean['mainproduct_numeric'] = pd.Categorical(df_clean['mainproduct']).codes
df_clean['year_numeric'] = pd.Categorical(df_clean['year']).codes

# Dependent variable: ln_sl_corr (already in log form)
# Convert to levels since replicate function logs internally
y = np.exp(df_clean['ln_sl_corr'])

# Main treatment variable: cohortage (firm age)
main_var = 'cohortage'

# X matrix: cohortage + year FE + mainproduct FE (absorbed)
X_vars = [main_var, 'year_numeric', 'mainproduct_numeric']
X = df_clean[X_vars].copy()

# Add constant
X = sm.add_constant(X, prepend=False)

# Get the indices of FE columns in X
# mainproduct is absorbed (industry FE), year is included as FE
fe_indices = [X.columns.get_loc('year_numeric'), X.columns.get_loc('mainproduct_numeric')]

# METADATA
metadata = {
    'paper_id': '111',
    'table_id': '2',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 2 Column 1: Basic markup lifecycle profile - ln(markup) on firm age with year and industry (mainproduct) fixed effects. Sample: 1991-2000, Sample==1 firms only. Expected N=55,212.'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=main_var,  # Variable of interest: cohortage
    fe=fe_indices,  # Year and industry fixed effects
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)

