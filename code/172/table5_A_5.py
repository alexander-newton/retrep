import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

# Path setup for your project replication helper
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
# Table V, Panel A, Col 5 spec:
# reg lrgdpch2010 tricksters_punish lnyear_firstpub lnnmbr_title, r
# DV is log(GDP pc 2010) in the file, but replicate() logs internally -> pass levels.
# -------------------------------
dv_log = 'lrgdpch2010'  # log in the file
main_x = 'tricksters_punish'
controls = ['lnyear_firstpub', 'lnnmbr_title']

needed = [dv_log, main_x] + controls
df_clean = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

# y in LEVELS (because replicate will log it internally)
y = np.exp(df_clean[dv_log].values)

# X: main regressor + controls + constant
X = df_clean[[main_x] + controls].copy()
X = sm.add_constant(X, prepend=False)

# No endogeneity, no instruments, no fixed effects for col 5
metadata = {
    'paper_id': '172',
    'table_id': '5',
    'panel_identifier': 'A_5',
    'model_type': 'log-linear',
    'comments': 'Table V Panel A Column 5: lrgdpch2010 on tricksters_punish with lnyear_firstpub and lnnmbr_title; robust SE.'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=[main_x],
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
