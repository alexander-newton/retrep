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
filepath = os.path.join(INPUT_DATA_DIR, '055/gravity_analysis_final.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 1 Panel A Column 2: IV (Second-stage)
# Dependent variable: uprob (probability of commuting)
# Endogenous variable: lttrans1921 (log transit travel time)
# Instrument: ltt1921 (log walking travel time)
# Fixed effects: Origin and Destination borough indicators
# Sample: sample==1 (positive commuting flows, excluding self-commuting)
# =====================================

# Apply sample restriction and drop missing values
df_sample = df[df['sample'] == 1].copy()
df_sample = df_sample.dropna(subset=['uprob', 'lttrans1921', 'ltt1921', 'G_UNIT_o', 'G_UNIT_d'])

# Dependent variable
y = df_sample['uprob'].values

# Create fixed effects as dummy variables
origin_dummies = pd.get_dummies(df_sample['G_UNIT_o'], prefix='origin', drop_first=True)
dest_dummies = pd.get_dummies(df_sample['G_UNIT_d'], prefix='dest', drop_first=True)

# X matrix: endogenous variable + fixed effects
X = pd.concat([
    df_sample[['lttrans1921']].reset_index(drop=True),
    origin_dummies.reset_index(drop=True),
    dest_dummies.reset_index(drop=True)
], axis=1)

# Z matrix: instrument + fixed effects  
z = pd.concat([
    df_sample[['ltt1921']].reset_index(drop=True),
    origin_dummies.reset_index(drop=True),
    dest_dummies.reset_index(drop=True)
], axis=1)

# Ensure all columns are numeric
X = X.astype(float)
z = z.astype(float)

# Add constant
X = sm.add_constant(X, prepend=False)
z = sm.add_constant(z, prepend=False)

# METADATA
metadata = {
    'paper_id': '055',
    'table_id': '1',
    'panel_identifier': 'A_2',
    'model_type': 'log-log',
    'comments': 'Table 1 Panel A Column 2: IV regression of log probability of commuting (uprob) on log transit travel time (lttrans1921) instrumented by log walking time (ltt1921) with origin and destination FE'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='lttrans1921',
    endog_x='lttrans1921',
    z=z,
    elasticity=True,
    fe=None,
    output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True

)