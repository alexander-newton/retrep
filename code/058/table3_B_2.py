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

df['lngoldprice_SB'] = df['lngoldprice'] * df['stationary_bandit']
df['lngoldprice_FARDC'] = df['lngoldprice'] * df['FARDC']
df['lngoldprice_SB_MILITIA'] = df['lngoldprice'] * df['MILITIA']

required_vars = ['marriages_', 'lngoldprice', 'lngoldprice_SB', 
                 'lngoldprice_FARDC', 'lngoldprice_SB_MILITIA', 
                 'year', 'groupidv']

df_clean = df[required_vars].dropna().reset_index(drop=True)

y_col2 = np.exp(df_clean['marriages_'].values)

treatment_vars = ['lngoldprice', 'lngoldprice_SB', 'lngoldprice_FARDC', 
                  'lngoldprice_SB_MILITIA']

fe_vars = ['year', 'groupidv']

X_col2 = df_clean[treatment_vars + fe_vars].copy()
X_col2 = sm.add_constant(X_col2, prepend=False)

cluster_col2 = df_clean['groupidv'].values

metadata_col2 = {
    'paper_id': '058',
    'table_id': '3',
    'panel_identifier': 'B_2',
    'model_type': 'log-log',
    'comments': 'Table 3 Panel B Column 2: log(weddings+1) with log gold price interactions; year and groupidv FE; clustered SE'
}

interest_var = 'lngoldprice_SB_MILITIA'

replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest=interest_var,
    elasticity=True,
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)