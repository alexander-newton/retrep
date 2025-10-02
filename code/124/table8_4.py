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

# Apply sample restrictions for column 4 (1998-2000 period)
sample_condition = (
    (df['urban'] == 1) & 
    (df['male'] == 1) & 
    (df['age'] > 15) & 
    (df['age'] < 80) & 
    (df['total'] > 1) & 
    (df['year'] > 1997)
)
df_clean = df[sample_condition].copy()

# Remove missing values for required variables
required_vars = ['lrttpayh', 'xr', 'edxroied', 'xroied', 'oilprice', 'agemsq', 
                'oilped', 'agemoilp', 'agemsqoilp', 'agemxroied', 'agemsqxroied', 'pidlink']
df_clean = df_clean.dropna(subset=required_vars)

# Dependent variable (in levels since replicate function logs internally)  
y = np.exp(df_clean['lrttpayh'].values)

# Apply scaling as noted in original table
# Exchange rate ×10⁻⁴, edxroied ×10⁻⁵, xroied ×10⁻⁴
df_clean_scaled = df_clean.copy()
df_clean_scaled['xr'] = df_clean_scaled['xr'] * 1e-4
df_clean_scaled['edxroied'] = df_clean_scaled['edxroied'] * 1e-5
df_clean_scaled['xroied'] = df_clean_scaled['xroied'] * 1e-4

# Independent variables (NOT including individual fixed effects - let replicate handle them)
# Order matches the original Stata command: xr edxroied xroied oilprice agemsq oilped agemoilp agemsqoilp agemxroied agemsqxroied
X_vars = ['oilped','edxroied','oilprice', 'xroied','xr','agemsq',  
          'agemoilp', 'agemsqoilp', 'agemxroied', 'agemsqxroied', 'pidlink']
X = df_clean_scaled[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# Individual fixed effects for clustering
cluster = pd.Categorical(df_clean_scaled['pidlink']).codes

metadata = {
    'paper_id': '124',
    'table_id': '8',
    'panel_identifier': '4',
    'model_type': 'log-linear',
    'comments': 'Table 8 Column 4: FE regression of log urban hourly wages (1998-2000), fixed effects and clustered by individual (pidlink)'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=['xr'],
    elasticity=False,
    fe=['pidlink'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
