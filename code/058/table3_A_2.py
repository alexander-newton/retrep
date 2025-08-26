import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

filepath = os.path.join(INPUT_DATA_DIR, '058/SITE ANALYSIS.dta')
df = pd.read_stata(filepath)

df = df[df['mine'] == 0].copy()

df['lncoltanprice_SB'] = df['lncoltanprice'] * df['stationary_bandit']
df.loc[df['FARDC'] == 1, 'lncoltanprice_SB'] = 0

df['lncoltanprice_SB_MILITIA'] = df['lncoltanprice'] * df['MILITIA']
df['lncoltanprice_FARDC'] = df['lncoltanprice'] * df['FARDC']

# Create binary airport variable (1 if close to airport, 0 otherwise)
# Try median split or check if there's already a binary indicator
if 'airport' in df.columns:
    df['lncoltanprice_airport'] = df['lncoltanprice'] * df['airport']
else:
    # Create binary: 1 if below median distance (closer to airport)
    df['airport_binary'] = (df['distance_airport_v'] <= df['distance_airport_v'].median()).astype(int)
    df['lncoltanprice_airport'] = df['lncoltanprice'] * df['airport_binary']

required_vars = ['marriages_', 'lncoltanprice', 'lncoltanprice_SB', 
                 'lncoltanprice_FARDC', 'lncoltanprice_SB_MILITIA', 
                 'lncoltanprice_airport', 'year', 'groupidv']

df_clean = df[required_vars].dropna().reset_index(drop=True)

y_col2 = np.exp(df_clean['marriages_'].values)

treatment_vars = ['lncoltanprice', 'lncoltanprice_SB', 'lncoltanprice_FARDC', 
                  'lncoltanprice_SB_MILITIA', 'lncoltanprice_airport']

fe_vars = ['year', 'groupidv']

X_col2 = df_clean[treatment_vars + fe_vars].copy()
X_col2 = sm.add_constant(X_col2, prepend=False)

cluster_col2 = df_clean['groupidv'].values

metadata_col2 = {
    'paper_id': '058',
    'table_id': '3',
    'panel_identifier': 'A_2',
    'model_type': 'log-log',
    'comments': 'Table 3 Panel A Column 2: log(weddings+1) with log coltan price interactions; year and groupidv FE; clustered SE'
}

interest_var = 'lncoltanprice_SB_MILITIA'

replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest=interest_var,
    elasticity=True,
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
)