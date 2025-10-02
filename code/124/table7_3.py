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
filepath = os.path.join(INPUT_DATA_DIR, '124/table7.dta')
df = pd.read_stata(filepath)

# Apply sample restrictions for column 3 (1993-2000 period, individual FE)
sample_condition = (
    (df['urban'] == 1) & 
    (df['male'] == 1) & 
    (df['age'] > 15) & 
    (df['age'] < 80) & 
    (df['total'] > 1) & 
    (df['year'] > 1992)
)
df_clean = df[sample_condition].copy()

# Create education-year interactions manually (edyrs * year dummies)
# Use 2000 as the base year (omitted category)
years = [1993, 1994, 1995, 1996, 1997, 1998, 1999]  # 2000 is the base
for yr in years:
    df_clean[f'edyrs_x_yr{yr}'] = df_clean['edyrs'] * (df_clean['year'] == yr).astype(int)

# Create year dummies (with 2000 as omitted base)
for yr in years + [2000]:
    df_clean[f'yr{yr}'] = (df_clean['year'] == yr).astype(int)

# Education-year interaction variables (manually created)
edyrsyr_vars = [f'edyrs_x_yr{yr}' for yr in years]
# Year dummy variables  
year_vars = [f'yr{yr}' for yr in years]  # 2000 omitted as base

# Remove missing values for required variables
required_vars = ['lrttpayh', 'agem', 'agemsq', 'edyrs', 'pidlink'] + year_vars + edyrsyr_vars
df_clean = df_clean.dropna(subset=required_vars)

# Dependent variable (in levels since replicate function logs internally)  
y = np.exp(df_clean['lrttpayh'].values)

# Independent variables for column 3 (FE with education-year interactions)
# Order education-year interactions first (as shown in table), then age controls, then year dummies + pidlink for FE
edyrsyr_ordered = sorted([var for var in edyrsyr_vars if var in df_clean.columns])
year_ordered = sorted([var for var in year_vars if var in df_clean.columns])
X_vars = edyrsyr_ordered + ['agem', 'agemsq'] + year_ordered + ['pidlink']
X = df_clean[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# Individual fixed effects for clustering
cluster = pd.Categorical(df_clean['pidlink']).codes

metadata = {
    'paper_id': '124',
    'table_id': '7',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'Table 7 Column 3: FE regression of log urban hourly wages (1993-2000), education-year interactions first, age controls, individual FE and clustered by pidlink'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=['agem'], # as fixed effects not implemented for multiple variables of interest
    fe=['pidlink'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
