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
filepath = os.path.join(INPUT_DATA_DIR, '066/Accounts_matched_collapsed.dta')
df = pd.read_stata(filepath)

# Use sales as outcome (in levels) - replicate function will log it
if 'sales' in df.columns:
    df['y'] = df['sales']
else:
    df['y'] = np.exp(df['ly'])  # Convert ly to levels if sales not available

# Noise controls
noise_controls = ['pa', 'reliability']
# Add ww* variables
ww_vars = [col for col in df.columns if col.startswith('ww')]
# Add aa* variables  
aa_vars = [col for col in df.columns if col.startswith('aa')]
noise_controls = noise_controls + ww_vars + aa_vars

# Remove missing values including noise controls
key_vars = ['y', 'ceo_behavior', 'lemp', 'lempm', 'cons', 'active', 
            'year', 'cty', 'emp_imputed', 'sic', 'r_averagewk'] + noise_controls
key_vars = [v for v in key_vars if v in df.columns]  # Only keep existing columns
df = df.dropna(subset=key_vars)

# Build X matrix - matching Stata specification exactly
X_vars = ['ceo_behavior', 'lemp', 'lempm', 'cons', 'active', 'emp_imputed'] + noise_controls

# Add fixed effects columns to X
fe_vars = ['year', 'cty', 'sic']
X_vars = X_vars + fe_vars

# Create X matrix with all variables including FEs
X = sm.add_constant(df[X_vars], prepend=False)

# Define weights for the regression
weights = df['r_averagewk'].values

# Metadata
metadata = {
    'paper_id': '066',
    'table_id': '3',
    'panel_identifier': '1',
    'model_type': 'log-linear'
}

# Run replication
replicate(
    metadata=metadata,
    y=df['y'].values,  # Pass as numpy array
    X=X,  # Keep as DataFrame for FE handling
    interest='ceo_behavior',
    fe=['year', 'cty', 'sic'],
    #weights=weights,
)