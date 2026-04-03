import numpy as np
import os
import sys
import json
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from replication import replicate
from utils.json_utils import ReplicationJSONBuilder, load_config

# Load configuration
config = load_config()
output_folder = config.get('output', './output')

# =====================================
# Paper 183 - "Intergenerational Mobility in Florence" (REStud)
# Table 3, Column 3, Panel A: Earnings
# reg lwage lwage_1427_sex_age sex age agesq [w=freq], r
# =====================================

# Load data
df = pd.read_stata('/Users/ellenmunroe/Downloads/replication package/restud_2011.dta')

# DV in levels
df['wage_2011'] = np.exp(df['lwage'])

# Drop missing
reg_df = df[['wage_2011', 'lwage_1427_sex_age', 'sex', 'age', 'agesq', 'freq']].dropna().reset_index(drop=True)

metadata = {
    'paper_id': '183',
    'table_id': 'Table3',
    'panel_identifier': 'Col3_PanelA',
    'model_type': 'log-log'
}

y = pd.DataFrame(reg_df['wage_2011'].values, columns=['wage_2011'])
X = reg_df[['lwage_1427_sex_age', 'sex', 'age', 'agesq', 'freq']].reset_index(drop=True)

out_dir = os.path.join(output_folder, '183', 'Table3_Col3')
os.makedirs(out_dir, exist_ok=True)
y.to_parquet(os.path.join(out_dir, 'y.parquet'), index=False)
X.to_parquet(os.path.join(out_dir, 'X.parquet'), index=False)
with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

replicate(
    metadata=metadata,
    y=y.values,
    X=X[['lwage_1427_sex_age', 'sex', 'age', 'agesq']].values,
    interest='lwage_1427_sex_age',
    endog_x=None,
    z=None,
    fe=None,
    elasticity=True,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols={'cov_type': 'HC1'},
    kwargs_ppml=None,
    fit_full_model=False,
    output=True,
    output_dir=output_folder
)
