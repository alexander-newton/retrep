import yaml
import os
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '156/LIA_main_dataset.dta')
df = pd.read_stata(filepath)

# ========= TABLE 2, COL (2) =========
# Spec: areg logcitysize yeartemp yeard* (historical_×period) (geographic_×period), absorb(cityid)
# replicate() logs y internally → pass y in LEVELS

# ---- Historical controls (× period) ----
hist_cols = [
    'Anglican1500','Anglican1600','Anglican1700','Anglican1750','Anglican1800','Anglican1850',
    'Luther1500','Luther1600','Luther1700','Luther1750','Luther1800','Luther1850',
    'Calcatlutf1500','Calcatlutf1600','Calcatlutf1700','Calcatlutf1750','Calcatlutf1800','Calcatlutf1850',
    'Recovcath1500','Recovcath1600','Recovcath1700','Recovcath1750','Recovcath1800','Recovcath1850',
    'Catholic1500','Catholic1600','Catholic1700','Catholic1750','Catholic1800','Catholic1850',
    'Calvhug1500','Calvhug1600','Calvhug1700','Calvhug1750','Calvhug1800','Calvhug1850',
    'university1500','university1600','university1700','university1750','university1800','university1850',
    'Atlantictrader1500','Atlantictrader1600','Atlantictrader1700','Atlantictrader1750','Atlantictrader1800','Atlantictrader1850',
    'RomanRoad1500','RomanRoad1600','RomanRoad1700','RomanRoad1750','RomanRoad1800','RomanRoad1850',
    'RomanEmpire1500','RomanEmpire1600','RomanEmpire1700','RomanEmpire1750','RomanEmpire1800','RomanEmpire1850',
    'lnoceandistance1500','lnoceandistance1600','lnoceandistance1700','lnoceandistance1750','lnoceandistance1800','lnoceandistance1850'
]

# ---- Geographic controls (× period) + rainyear ----
geo_cols = [
    'lnGAEZaltitude1500','lnGAEZaltitude1600','lnGAEZaltitude1700','lnGAEZaltitude1750','lnGAEZaltitude1800','lnGAEZaltitude1850',
    'lnwheatindex1500','lnwheatindex1600','lnwheatindex1700','lnwheatindex1750','lnwheatindex1800','lnwheatindex1850',
    'lnpotatoindex1500','lnpotatoindex1600','lnpotatoindex1700','lnpotatoindex1750','lnpotatoindex1800','lnpotatoindex1850',
    'lnruggedness1500','lnruggedness1600','lnruggedness1700','lnruggedness1750','lnruggedness1800','lnruggedness1850',
    'rainyear'
]

# ---- Year FE ----
year_fe_cols = ['yeard1','yeard2','yeard3','yeard4','yeard5','yeard6']

# Build X and y
X_base_cols = ['yeartemp'] + hist_cols + geo_cols + year_fe_cols
# ↓↓↓ added .dropna() here ↓↓↓
df_clean = df[['logcitysize','cityid','year'] + X_base_cols].dropna().copy()

# FE encodings
df_clean['cityid_numeric'] = pd.Categorical(df_clean['cityid']).codes
# df_clean['year_numeric'] = pd.Categorical(df_clean['year']).codes  # not needed; year FE are yeard1–yeard6

# y in LEVELS (replicate logs internally)
y = np.exp(df_clean['logcitysize'])

# X with FE placeholders
X_vars = X_base_cols + ['cityid_numeric']
X = df_clean[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# FE indices (positions inside X)
fe_indices = [X.columns.get_loc('cityid_numeric')]

# Metadata
metadata = {
    'paper_id': '156',
    'table_id': '2',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comments': 'Table 2 Column 2: log(city size) on yeartemp with city FE, year FE, and full period-interacted historical & geographic controls. No clustering.'
}

# Run replication (no clustering)
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='yeartemp',
    fe=fe_indices,
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
