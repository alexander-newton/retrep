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
filepath = os.path.join(INPUT_DATA_DIR, '124/table8.dta')
df = pd.read_stata(filepath)

# Apply sample restrictions for column 2 (1993-2000 period)
sample_condition = (
    (df['urban'] == 1) & 
    (df['male'] == 1) & 
    (df['age'] > 15) & 
    (df['age'] < 80) & 
    (df['total'] > 1) & 
    (df['move'] == 0)
)
df_clean = df[sample_condition].copy()

# Remove missing values for required variables
required_vars = ['oilped', 'oilprice', 'lrttpay', 'agemsq', 'year', 'agemoilp', 'agemsqoilp', 'pidlink']
df_clean = df_clean.dropna(subset=required_vars)

# Dependent variable (in levels since replicate function logs internally)
y = np.exp(df_clean['lrttpay'].values)

# Independent variables (NOT including individual fixed effects - let replicate handle them)
X_vars = ['oilped','oilprice','agemsq', 'year', 'agemoilp', 'agemsqoilp', 'pidlink']
X = df_clean[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# Individual fixed effects for clustering
cluster = pd.Categorical(df_clean['pidlink']).codes

metadata = {
    'paper_id': '124',
    'table_id': '8',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comments': 'Table 8 Column 2: FE regression of log urban hourly wages (1993-2000), fixed effects by individual (pidlink)'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=['oilprice'],
    fe=['pidlink'],
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
