import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -----------------------
# Load configuration
# -----------------------
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# -----------------------
# LOAD DATA (policy-cycle CSV, no header)
# Primary: <INPUT_DATA_DIR>/086/data_policycle.csv
# Fallback: /mnt/data/data_policycle.csv (if running locally with uploaded file)
# -----------------------
primary_path = os.path.join(INPUT_DATA_DIR, '086', 'data_policycle.csv')
fallback_path = '/mnt/data/data_policycle.csv'
filepath = primary_path if os.path.exists(primary_path) else fallback_path

raw = pd.read_csv(filepath, header=None)

# MATLAB column mapping (1-based) -> pandas 0-based:
# date=1 -> 0, dem_dum4=6 -> 5, vwretd=8 -> 7, rf=11 -> 10
col_map = {'date': 0, 'dem_dum4': 5, 'vwretd': 7, 'rf': 10}
df = raw.iloc[:, [col_map['date'], col_map['dem_dum4'], col_map['vwretd'], col_map['rf']]].copy()
df.columns = ['date', 'dem_dum4', 'vwretd', 'rf']

# -----------------------
# CLEAN + SAMPLE WINDOW
# -----------------------
# Treat -99 as missing (author convention), enforce numeric, drop NA
df.replace(-99, np.nan, inplace=True)
df['date'] = pd.to_numeric(df['date'], errors='coerce')
df['vwretd'] = pd.to_numeric(df['vwretd'], errors='coerce')
df['rf'] = pd.to_numeric(df['rf'], errors='coerce')
df['dem_dum4'] = pd.to_numeric(df['dem_dum4'], errors='coerce')

df.dropna(subset=['date', 'dem_dum4', 'vwretd', 'rf'], inplace=True)

# Sample: 1927:01–2015:12
df = df[(df['date'] >= 192701) & (df['date'] <= 201512)].copy()

# -----------------------
# Construct DV as a RATIO (replicate() logs internally)
# y = (1 + R_mkt) / (1 + R_rf)
# Guard against log(<=0) later by filtering
# -----------------------
one_plus_mkt = 1.0 + df['vwretd'].to_numpy()
one_plus_rf  = 1.0 + df['rf'].to_numpy()
valid = (one_plus_mkt > 0) & (one_plus_rf > 0)
df = df.loc[valid].copy()

df['y_ratio'] = (1.0 + df['vwretd']) / (1.0 + df['rf'])
df['Dem'] = df['dem_dum4'].astype(int)

# -----------------------
# DESIGN: Reduced form — constant + Dem dummy
# -----------------------
y = df['y_ratio']                                # replicate() will take logs internally
X = sm.add_constant(df[['Dem']], has_constant='add')

# -----------------------
# METADATA
# -----------------------
metadata = {
    'paper_id': '086',
    'table_id': '1',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'Table 1 row 1 col 3: Democrat–Republican difference. DV is ratio (1+R_mkt)/(1+R_rf); monthly sample 1927:01–2015:12.'
}

# -----------------------
# RUN THE REPLICATION
# coef on 'Dem' = monthly Dem–Rep difference (log scale internally)
# Annualize later if your replicate() prints monthly units; paper reports x1200
# -----------------------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='Dem',
    elasticity=False,
    fe=None,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)


print("\nNote: The reported coefficient is monthly.")
print("To match Table 1, multiply the Dem coefficient by 1200 to get the annualized % difference.")
print("Additional rows in Table 1 (subsamples) are robustness checks;")
print("replicating only the full-sample row (1927:01–2015:12) is enough for the main result.")