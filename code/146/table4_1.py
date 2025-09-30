# code/146/table5_1.py
import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ---------------------------
# Load configuration
# ---------------------------
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ---------------------------
# LOAD DATA
# ---------------------------
filepath = os.path.join(INPUT_DATA_DIR, '146/Cage_Herve_Viaud_Restud_dataset.dta')
df = pd.read_stata(filepath)

# ---------------------------
# SAMPLE RESTRICTIONS (match Stata)
# drop if doc_date == "2012.12.31"
# drop AFP/Reuters (id_news == 5001/5002)
# scale length vars to 'thsd ch'
# ---------------------------
df['doc_date'] = df['doc_date'].astype(str)
df = df[df['doc_date'] != "2012.12.31"]
df = df[~df['id_news'].isin([5001, 5002])]

for v in ['doc_size', 'original_content', 'NO_content']:
    if v in df.columns:
        df[v] = df[v] / 1000.0

# Ensure ln DV exists; if not, build from levels
if 'ln_nb_facebook_share' not in df.columns:
    df['ln_nb_facebook_share'] = np.log1p(df['nb_facebook_share'])

# ---------------------------
# DEP VAR
# replicate() logs internally, so pass levels = exp(ln)
# ln_nb_facebook_share = log(1 + shares)  -> y = 1 + shares
# ---------------------------
y = np.exp(df['ln_nb_facebook_share'].astype(float).values)

# ---------------------------
# X: main regressors for Table 5 col (1)
#   doc_rank, doc_reactivity, doc_originality, doc_size
# FE: id_news, doc_date, event_id  (as dummies)
# ---------------------------
main_vars = ['doc_rank', 'doc_reactivity', 'doc_originality', 'doc_size']
X_main = df[main_vars].astype(float)

news_fe   = pd.get_dummies(df['id_news'].astype('category'),   prefix='news',  drop_first=True)
date_fe   = pd.get_dummies(df['doc_date'].astype('category'),  prefix='date',  drop_first=True)
event_fe  = pd.get_dummies(df['event_id'].astype('category'),  prefix='event', drop_first=True)

X = pd.concat([X_main, news_fe, date_fe, event_fe], axis=1)
X = sm.add_constant(X, prepend=False)

# FE var names for metadata/table footer (match your pattern)
fe_vars = news_fe.columns.tolist() + date_fe.columns.tolist() + event_fe.columns.tolist()

# Align & drop missing
required_cols = main_vars + fe_vars + ['const']
mask = np.isfinite(y) & X[required_cols].apply(np.isfinite).all(axis=1)
y = y[mask]
X = X.loc[mask, :]

# ---------------------------
# METADATA
# ---------------------------
metadata = {
    'paper_id': '146',
    'table_id': '5',
    'panel_identifier': '1',  # Table 5, Column (1)
    'model_type': 'log-linear',
    'comments': (
        'Table 5, col (1): ln(Facebook shares) on doc_rank, doc_reactivity, '
        'doc_originality, doc_size with media (id_news), date (doc_date), and '
        'event (event_id) fixed effects. DV passed in levels (exp(ln)) so '
        'replicate() can log internally.'
    )
}

# ---------------------------
# RUN REPLICATION
# ---------------------------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='doc_originality',
    elasticity=False,
    fe=fe_vars,                    # << like your example

)
