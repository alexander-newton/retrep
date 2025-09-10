import os
import yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ----------------------------
# Load config
# ----------------------------
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ----------------------------
# Load data
# ----------------------------
filepath = os.path.join(INPUT_DATA_DIR, '099/obs_rep.dta')
df = pd.read_stata(filepath)


# ----------------------------
# Prepare variables (Table 5, Col 4)
# ----------------------------
# Dependent variable: lcsnm is already in logs, convert to levels
y = np.exp(df['lcsnm'].values)

# Main regressor: observed population diversity
main_treatment = 'odiv'

# Controls by blocks (matching Stata specification):
# ${regions} = i.cont ssa
# ${base} = abslat rugg elev elev_range postcal postcal_range island distwater  
# ${climate} = tmp pre
# frac = ethnolinguistic fractionalization

controls = [
    # Region controls
        # Fractionalization
    'frac',
    'ssa',  # ssa dummy (continent FE will be handled separately)
    # Base geographic controls
    'abslat', 'rugg', 'elev', 'elev_range',
    'postcal', 'postcal_range', 'island', 'distwater',
    # Climate controls  
    'tmp', 'pre'

]

# Fixed effects: continent (cont)
fe_variables = ['cont']

# Build design matrix
X_data = df[[main_treatment] + controls + fe_variables].copy()
X = sm.add_constant(X_data, prepend=False)

# ----------------------------
# Metadata (for your tracking)
# ----------------------------
metadata_c4 = {
    'paper_id': '099',
    'table_id': '5',
    'panel_identifier': '4',
    'model_type': 'log-linear',
    'comments': 'Replication of Table 5, Column (4): OLS with continent FE, base+climate controls, and ethnolinguistic fractionalization.'
}

# ----------------------------
# Save outputs
# ----------------------------
replicate(
    metadata=metadata_c4,
    y=y,
    X=X,
    interest=0,  # Index of odiv in X (first column, index 0)
    elasticity=False,
    fe=fe_variables,  # Pass continent column names for fixed effects
    output=True, 
    output_dir=OUTPUT_DIR, 
    replicated=True, 
)
