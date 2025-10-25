import os, sys, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Import replicate()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -----------------------------
# Config + fixed filepath
# -----------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

csv_path = os.path.join(
    INPUT_DATA_DIR, '065/SEcorrectiondata_removedomfirm_st_180823.csv'
)
df = pd.read_csv(csv_path, engine='python')


# ---------- VARIABLES ----------
fare_log   = df['lroutefare']             # log(price)
quarter    = df['quarter'].astype(int)    # FE code
phat       = df['phat']
control    = df['control']                # SWPE (phase-2)
fuel       = df['fuel']
fueldist   = df['fueldist']
deviation2 = df['deviationsq']
comp_d     = df['competitor_d']
comp_i     = df['competitor_i']
PE_ctrl    = df['PotEntrants_AND']        # PE control

# key regressors
probs  = phat * control                   # Pr(entry) × SWPE
phatsq = probs ** 2                       # [Pr(entry) × SWPE]^2

# compress MFE_* one-hots to a single market FE code
mfe_cols = [c for c in df.columns if c.startswith('MFE_')]
if not mfe_cols:
    raise KeyError("No MFE_* columns found for market FE.")
marketFE = df[mfe_cols].to_numpy()
market_idx = marketFE.argmax(axis=1) + 1
market_has = (marketFE.max(axis=1) > 0)
fe_market = pd.Series(np.where(market_has, market_idx, 0), name='fe_market').astype(int)

# ---------- SAMPLE RULES ----------
# 1) MATLAB rule: drop tiny log(price)
dummy1 = 0.01
keep = (fare_log >= dummy1)

# ---------- BUILD JOINT MODEL DF & DROP MISSINGS (final N≈3884) ----------
df_model = pd.DataFrame({
    'y':           np.exp(fare_log[keep]),   # levels; replicate logs internally
    'fuel':        fuel[keep],
    'fueldist':    fueldist[keep],
    'deviationsq': deviation2[keep],
    'probs':       probs[keep],
    'phatsq':      phatsq[keep],
    'control':     control[keep],            # SWPE
    'competitor_d': comp_d[keep],
    'competitor_i': comp_i[keep],
    'PE':          PE_ctrl[keep],
    'fe_quarter':  quarter[keep],
    'fe_market':   fe_market[keep]
})

# drop any missing/infinite to align y & X exactly
df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
print(f"Final sample size after dropping missing: {len(df_model)}")

# ---------- SPLIT TO y, X & FE ----------
y = df_model.pop('y').to_numpy()
X = sm.add_constant(df_model, prepend=False)

# identify FE columns for replicate()
fe_indices = [X.columns.get_loc('fe_quarter'), X.columns.get_loc('fe_market')]

# ---------- METADATA ----------
meta_base = {
    'paper_id': '065',
    'table_id': '3',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 3, Col (1): log(price) on SWPE, Pr(entry)×SWPE, [Pr(entry)×SWPE]^2; y in levels; FE: quarter & market; rows with missing dropped.'
}

# ---------- RUN: three replicate() calls (one per reported coef) ----------
replicate(
    metadata=meta_base,
    y=y, X=X, interest='probs', fe=fe_indices,
    elasticity=False, output=True, output_dir=OUTPUT_DIR, replicated=True
)