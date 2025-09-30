import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ==============================
# Config
# ==============================
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ==============================
# Load data
# ==============================
filepath = os.path.join(INPUT_DATA_DIR, '146/Cage_Herve_Viaud_Restud_dataset.dta')
df = pd.read_stata(filepath)

# ==============================
# Sample restrictions (Stata-consistent)
# ==============================
df['doc_date'] = df['doc_date'].astype(str)
df = df[df['doc_date'] != "2012.12.31"]
df = df[~df['id_news'].isin([5001, 5002])]  # drop AFP/Reuters

# Length in thousands of characters
for v in ['doc_size', 'original_content', 'NO_content']:
    df[v] = df[v] / 1000.0

# ==============================
# Drop missing (tight)
# ==============================
main_vars = ['doc_rank', 'doc_reactivity', 'doc_originality', 'doc_size']
required_vars = main_vars + ['ln_nb_facebook_share', 'id_news', 'doc_date', 'event_id']
df = df.dropna(subset=required_vars)

# ==============================
# Dependent variable (levels; replicate() logs internally)
# ==============================
y = np.exp(df['ln_nb_facebook_share'].values)  # = 1 + shares

# ==============================
# X: regressors + FE
# ==============================
X_main = df[main_vars].astype(float)
news_fe  = pd.get_dummies(df['id_news'].astype('category'),  prefix='news',  drop_first=True)
date_fe  = pd.get_dummies(df['doc_date'].astype('category'), prefix='date',  drop_first=True)
event_fe = pd.get_dummies(df['event_id'].astype('category'), prefix='event', drop_first=True)

X = pd.concat([X_main, news_fe, date_fe, event_fe], axis=1)
X = sm.add_constant(X, prepend=False)

fe_vars = news_fe.columns.tolist() + date_fe.columns.tolist() + event_fe.columns.tolist()

# ==============================
# Metadata
# ==============================
metadata = {
    'paper_id': '146',
    'table_id': '5',
    'panel_identifier': '1',   # Table 5, Column (1)
    'model_type': 'log-linear',
    'comments': 'Table 5 Col (1): ln(FB shares) ~ doc_rank + doc_reactivity + doc_originality + doc_size + FE(id_news, doc_date, event_id). DV passed in levels for internal log.'
}

# ==============================
# Run
# ==============================
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='doc_originality',
    elasticity=False,
    fe=fe_vars,

)
