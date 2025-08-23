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
filepath = os.path.join(INPUT_DATA_DIR, '051/CAST_data2.dta')
df = pd.read_stata(filepath)

# Create deposits in levels BEFORE cleaning
df['dep'] = np.exp(df['logdep'])

# Year-interacted controls
all_controls = []
control_vars = ['rural_adult_91', 'log_y_pc_r_91', 'log_pop_area_91', 'alpha_adult_91']
for var in control_vars:
    for year in range(2, 16):  # y2 to y15
        col_name = f'{var}_y{year}'
        if col_name in df.columns:
            all_controls.append(col_name)

# Clean data - now include 'dep' in regression_vars
regression_vars = ['dep', 'logAsoy', 'AMC', 'year'] + all_controls
df_clean = df[regression_vars].dropna()

# Prepare data for regression
y = df_clean['dep'].values  # Now dep exists in df_clean
X = df_clean[['logAsoy'] + all_controls + ['AMC', 'year']]
X = sm.add_constant(X)

# For clustering
cluster = pd.Categorical(df_clean['AMC']).codes

metadata = {
    'paper_id': '051',
    'table_id': '3',
    'panel_identifier': '2',
    'model_type': 'log-log',
    'comments': 'Table III Column 2: Log deposits on log soy area with all controls, municipality and year FE, clustered SE at municipality level'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='logAsoy',
    elasticity=False,
    fe=['AMC', 'year'],
    output=True, output_dir=OUTPUT_DIR, replicated=True
)