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
# Paper 182 - Nunn & Puga (2012) "Ruggedness: The Blessing of Bad Geography in Africa"
# Table 2, Column 1: Omit 10 most rugged countries
# reg ln_rgdppc_2000 rugged rugged_x_africa cont_africa [stdcontrols] if mostrugged==0, robust
# =====================================

# Load data
df = pd.read_stata('/Users/ellenmunroe/Downloads/rugged_data/rugged_data.dta')

# Construct variables (matching do file)
df['ln_rgdppc_2000'] = np.log(df['rgdppc_2000'])
df['diamonds'] = df['gemstones'] / (df['land_area'] / 100)
df['rugged_x_africa'] = df['rugged'] * df['cont_africa']
for v in ['soil', 'tropical', 'dist_coast', 'diamonds']:
    df[f'{v}_x_africa'] = df[v] * df['cont_africa']

# Drop missing DV (needed for ranking)
df = df.dropna(subset=['ln_rgdppc_2000']).copy()

# Rank by ruggedness, flag 10 most rugged
df = df.sort_values('rugged').reset_index(drop=True)
df['rank'] = range(1, len(df) + 1)
df['mostrugged'] = (df['rank'] <= 10).astype(int)

# Drop 10 least rugged (sorted ascending, so rank 1-10 = least rugged)
df = df[df['mostrugged'] == 0].copy()

# Variables
controls = ['rugged', 'rugged_x_africa', 'cont_africa',
            'diamonds', 'diamonds_x_africa', 'soil', 'soil_x_africa',
            'tropical', 'tropical_x_africa', 'dist_coast', 'dist_coast_x_africa']

# Drop missing
sample_vars = ['rgdppc_2000'] + controls
reg_df = df[sample_vars].dropna().reset_index(drop=True)

# =====================================
# Table 2, Column 1
# =====================================

metadata = {
    'paper_id': '182',
    'table_id': 'Table2',
    'panel_identifier': 'Col1',
    'model_type': 'log-linear'
}

# y in levels (GDP per capita)
y = pd.DataFrame(reg_df['rgdppc_2000'].values, columns=['rgdppc_2000'])

# X: interest variable first (rugged), then rest (no FE in this regression)
X = reg_df[controls].reset_index(drop=True)

# Save to parquet
out_dir = os.path.join(output_folder, '182', 'Table2_Col1')
os.makedirs(out_dir, exist_ok=True)

y.to_parquet(os.path.join(out_dir, 'y.parquet'), index=False)
X.to_parquet(os.path.join(out_dir, 'X.parquet'), index=False)

with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# replicate()
replicate(
    metadata=metadata,
    y=y.values,
    X=X.values,
    interest='rugged',
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
