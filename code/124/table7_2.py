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

# Apply sample restrictions for column 2 (1993-2000 period, noconstant)
sample_condition = (
    (df['urban'] == 1) & 
    (df['male'] == 1) & 
    (df['age'] > 15) & 
    (df['age'] < 80) & 
    (df['total'] > 1) & 
    (df['year'] > 1992)
)
df_clean = df[sample_condition].copy()

# Get year dummy variables (yr* variables)
year_vars = [col for col in df_clean.columns if col.startswith('yr') and len(col) <= 4]
# Get education-year interaction variables (edyrsyr* variables) 
edyrsyr_vars = [col for col in df_clean.columns if col.startswith('edyrsyr')]

# Remove missing values for required variables
required_vars = ['lrttpayh', 'agem', 'agemsq', 'pidlink'] + year_vars + edyrsyr_vars
df_clean = df_clean.dropna(subset=[var for var in required_vars if var in df_clean.columns])

# Dependent variable (in levels since replicate function logs internally)  
y = np.exp(df_clean['lrttpayh'].values)

# Independent variables for column 2 (OLS with education-year interactions, noconstant)
# Order education-year interactions first (as shown in table), then age controls, then year dummies
edyrsyr_ordered = sorted([var for var in edyrsyr_vars if var in df_clean.columns])
year_ordered = sorted([var for var in year_vars if var in df_clean.columns])
X_vars = edyrsyr_ordered + ['agem', 'agemsq'] + year_ordered
X = df_clean[X_vars].copy()
# noconstant - don't add constant

# Individual fixed effects for clustering
cluster = pd.Categorical(df_clean['pidlink']).codes

metadata = {
    'paper_id': '124',
    'table_id': '7',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comments': 'Table 7 Column 2: OLS regression of log urban hourly wages (1993-2000), education-year interactions first, age controls, year dummies, clustered by individual (pidlink), noconstant'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=edyrsyr_ordered,
    fe=None,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
