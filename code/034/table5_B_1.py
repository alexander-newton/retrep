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
DATA_DIR = config['intermediatedata']

# Load data
filepath = os.path.join(DATA_DIR, '034/household_panel_final.dta')
df = pd.read_stata(filepath)

# Control variables
control_vars = [
    'b_head_age',
    'b_imp_head_age_dum',
    'b_head_female',
    'b_did_ganyu',
    'b_plan_ganyu',
    'b_acres_maize_total',
    'b_acres_cash_crops', 
    'b_harvest_total_value',
    'b_crop_diversity',
    'b_asset_quintile',
    'b_livestock_value',
    'b_input_value',
    'b_hired_ganyu',
    'b_num_farm_workers',
    'b_num_iga_workers',
    'control_gift'
]

# Add household member variables
member_vars = [col for col in df.columns if col.startswith('b_members_')]
control_vars.extend(member_vars)

# Block fixed effects
block_vars = [
    'block_dum_Chanje',
    'block_dum_Chiparamba', 
    'block_dum_Eastern',
    'block_dum_Southern',
    'block_dum_Western'
]

# Filter for Year 2 - Panel B uses year == 2
df_year2 = df[df['year'] == 2].copy()

# All controls
all_controls = control_vars + block_vars
existing_controls = [var for var in all_controls if var in df_year2.columns]

# Prepare data - make a proper copy
df_clean = df_year2.dropna(subset=['harvest_value', 'treated', 'treatedin1']).copy()

# Remove observations with zero or negative harvest_value
df_clean = df_clean[df_clean['harvest_value'] > 0].copy()

# Create interaction term (treated##treatedin1 creates this)
df_clean['treated_x_treatedin1'] = df_clean['treated'] * df_clean['treatedin1']

# Convert categorical/string columns to numeric
for col in ['treated', 'treatedin1', 'treated_x_treatedin1'] + existing_controls:
    if col in df_clean.columns:
        if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
            if len(df_clean[col].unique()) == 2:
                df_clean.loc[:, col] = pd.get_dummies(df_clean[col], drop_first=True).values.flatten()
            else:
                df_clean.loc[:, col] = pd.Categorical(df_clean[col]).codes

# Define y and X
# Panel B: reg l_harvest_value i.treated##i.treatedin1 i.year $controls $blocks if year == 2
y = df_clean['harvest_value'].values.astype(np.float64)
X = df_clean[['treated', 'treatedin1', 'treated_x_treatedin1'] + existing_controls].astype(np.float64)
X = sm.add_constant(X, prepend=False)

# Cluster variable
cluster = df_clean['vid'].values if 'vid' in df_clean.columns else None

# Metadata
metadata_B1 = {
    'paper_id': '034',
    'table_id': '5',
    'panel_identifier': 'B_1',
    'model_type': 'log-linear',
    'comments': 'Table 5 Panel B Column 1: Log harvest value with treated##treatedin1 interaction; Year 2'
}

# Run replication
replicate(
    metadata=metadata_B1,
    y=y,
    X=X,
    interest=['treated'],
    elasticity=False,
    fe=block_vars if all(var in existing_controls for var in block_vars) else None,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)