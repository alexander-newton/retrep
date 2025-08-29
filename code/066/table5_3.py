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

# Load data
filepath_ba = os.path.join(INPUT_DATA_DIR, '066/Accounts_matched_beforeafter.dta')
df_ba = pd.read_stata(filepath_ba)

# Define noise controls
noise_controls = ['pa', 'reliability']
noise_controls += [col for col in df_ba.columns if col.startswith('ww')]
noise_controls += [col for col in df_ba.columns if col.startswith('aa')]

# Collapse by cty, company_id, sic, after
agg_dict = {
    'emp_imputed': 'max',
    'ly': 'mean',
    'ceo_behavior': 'mean',
    'lemp': 'mean',
    'cons': 'mean',
    'active': 'mean',
    'year': 'mean',
    'r_averagewk': 'mean'
}
for col in noise_controls:
    if col in df_ba.columns:
        agg_dict[col] = 'mean'

df = df_ba.groupby(['cty', 'company_id', 'sic', 'after']).agg(agg_dict).reset_index()

# Process variables
df['year'] = df['year'].astype(int)
df['lemp'] = df['lemp'].fillna(-99)
df['lempm'] = (df['lemp'] == -99).astype(float)
df['ceo_behavior_after'] = df['ceo_behavior'] * df['after']
df['y'] = np.exp(df['ly'])

# Remove missing
key_vars = ['y', 'ceo_behavior', 'after', 'lemp', 'lempm', 'cons', 'active', 
            'year', 'cty', 'emp_imputed', 'company_id', 'r_averagewk'] + noise_controls
df = df.dropna(subset=[v for v in key_vars if v in df.columns])

# Build X matrix
X_vars = ['ceo_behavior', 'after', 'ceo_behavior_after', 'lemp', 'lempm', 
          'cons', 'active', 'emp_imputed'] + [col for col in noise_controls if col in df.columns]
X_vars += ['year', 'cty', 'company_id']
X = sm.add_constant(df[X_vars], prepend=False)

# Metadata
metadata = {
    'paper_id': '066',
    'table_id': '5',
    'panel_identifier': '3',
    'model_type': 'log-linear'
}

# Run replication
replicate(
    metadata=metadata,
    y=df['y'].values,
    X=X,
    interest=['after'],
    fe=['year', 'cty', 'company_id'],
    weights=df['r_averagewk'].values,
    output=True, output_dir=OUTPUT_DIR, replicated=True

)