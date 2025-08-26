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

required_vars = ['migrants_number', 'MILITIA', 'EXTERNAL_NOFARDC', 'FARDC', 
                 'lngoldprice', 'lncoltanprice', 'year', 'groupidv']

df_clean = df[required_vars].dropna().reset_index(drop=True)

# migrants_number is already log(1+migrants), so exp it to get 1+migrants
y_col3 = np.exp(df_clean['migrants_number'].values)

treatment_vars = ['MILITIA', 'EXTERNAL_NOFARDC', 'FARDC', 'lngoldprice', 'lncoltanprice']

fe_vars = ['year', 'groupidv']

X_col3 = df_clean[treatment_vars + fe_vars].copy()
X_col3 = sm.add_constant(X_col3, prepend=False)

cluster_col3 = df_clean['groupidv'].values

metadata_col3 = {
    'paper_id': '058',
    'table_id': '4',
    'panel_identifier': 'A_3',
    'model_type': 'log-linear',
    'comments': 'Table 4 Panel A Column 3: OLS - log(migrants+1) on armed group binary indicators (Militia, External, Army) with log price controls; year and municipality FE; clustered SE'
}

interest_vars = ['MILITIA', 'EXTERNAL_NOFARDC', 'FARDC']

replicate(
    metadata=metadata_col3,
    y=y_col3,
    X=X_col3,
    interest='MILITIA',
    elasticity=False,  # False because main IVs are binary, not logs
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col3}},
    output=True, output_dir=OUTPUT_DIR, replicated=True

)