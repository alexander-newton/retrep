import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

# Path setup for your replication helper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Load config
# -------------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# -------------------------------
# Load data
# -------------------------------
filepath = os.path.join(INPUT_DATA_DIR, '172/Country_Regressions_Ready.dta')
df = pd.read_stata(filepath)

# -------------------------------
# Table VI, Col 6 specification
# -------------------------------
dv_log = 'lnp06_18pc'          # log patents per capita (2006–18)
main_x = 'dnotsuccess'         # folklore share of non-successful challenges
controls = ['lnyear_firstpub', 'lnnmbr_title', 'asia', 'africa', 'americas', 'europe']

needed = [dv_log, main_x] + controls
df_clean = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

# replicate() logs DV internally → pass in levels
y = np.exp(df_clean[dv_log].values)

X = df_clean[[main_x] + controls].copy()
X = sm.add_constant(X, prepend=False)

metadata = {
    'paper_id': '172',
    'table_id': '6',
    'panel_identifier': 'A_6',
    'model_type': 'log-linear',
    'comments': 'Table VI Column 6: log patents per capita on dnotsuccess with publication controls + continent FEs'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=[main_x],
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True

)
