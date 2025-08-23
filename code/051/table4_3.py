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

# Create loans in levels BEFORE cleaning
df['loans'] = np.exp(df['logloans'])

# Year-interacted controls (all_controls_y)
all_controls = []
control_vars = ['rural_adult_91', 'log_y_pc_r_91', 'log_pop_area_91', 'alpha_adult_91']
for var in control_vars:
    for year in range(2, 16):  # y2 to y15
        col_name = f'{var}_y{year}'
        if col_name in df.columns:
            all_controls.append(col_name)

# Filter for non-soy producing municipalities (soy==0)
df_nonsoy = df[df['soy'] == 0].copy()

# Clean data
regression_vars = ['loans', 'municexp', 'AMC', 'year'] + all_controls
df_clean = df_nonsoy[regression_vars].dropna()

# Prepare data for regression
y = df_clean['loans'].values
X = df_clean[['municexp'] + all_controls + ['AMC', 'year']]
X = sm.add_constant(X)

# Use AMC values directly for clustering
cluster = df_clean['AMC'].values

metadata = {
    'paper_id': '051',
    'table_id': '4',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'Table IV Column 3: Log loans on municipality exposure for non-soy producing municipalities, all controls, municipality and year FE'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='municexp',  # Municipality exposure is the main variable
    elasticity=False,
    fe=['AMC', 'year'],
    output=True, output_dir=OUTPUT_DIR, replicated=True

)