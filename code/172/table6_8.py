import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

# Replication helper path
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
# Load data (Folder 172)
# -------------------------------
filepath = os.path.join(INPUT_DATA_DIR, '172/Country_Regressions_Ready.dta')
df = pd.read_stata(filepath)

# -------------------------------
# Table VI, Column 8 spec
# Stata: reg lnavgy06_18 dnotsuccess lnyear_firstpub lnnmbr_title asia africa americas europe, r
# DV is logged in file; replicate() logs internally → pass levels (exp of log)
# -------------------------------
dv_log = 'lnavgy06_18'  # log average new business registrations (2006–2018)
main_x = 'dnotsuccess'
controls = ['lnyear_firstpub', 'lnnmbr_title', 'asia', 'africa', 'americas', 'europe']

needed = [dv_log, main_x] + controls
df_clean = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

# y in LEVELS because replicate() will log internally
y = np.exp(df_clean[dv_log].values)

X = df_clean[[main_x] + controls].copy()
X = sm.add_constant(X, prepend=False)

metadata = {
    'paper_id': '172',
    'table_id': '6',
    'panel_identifier': 'A_8',
    'model_type': 'log-linear',
    'comments': 'Table VI Column 8: log new business registrations on dnotsuccess with publication controls + continent FEs'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=[main_x],
    elasticity=False,
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
