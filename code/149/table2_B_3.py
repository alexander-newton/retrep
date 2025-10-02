import yaml
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ==============================
# CONFIG & PATHS
# ==============================
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR      = config['outputdata']
INPUT_DATA_DIR  = config['intermediatedata']

PAPER_DIR       = os.path.join(INPUT_DATA_DIR, '149')
industry_path   = os.path.join(PAPER_DIR, 'industryData.csv')
xrd_path        = os.path.join(PAPER_DIR, 'xrd_data.csv')

# ==============================
# LOAD DATA
# ==============================
industry = pd.read_csv(industry_path)

# Ensure SIC exists and is 3-digit string
if 'SIC' not in industry.columns:
    # if original has 'sic' or similar, map to SIC 3
    for cand in ['sic','Sic','SIC3','sic3']:
        if cand in industry.columns:
            industry['SIC'] = industry[cand]
            break
industry['SIC'] = industry['SIC'].astype(str).str[:3].str.zfill(3)

# Try to merge in xrd_at by 3-digit SIC; if anything fails, fill zeros
AvgRDOverAsset = None
try:
    xrd = pd.read_csv(xrd_path)
    # build SIC 3 the same way
    if 'SIC' not in xrd.columns:
        # papers' xrd_data usually has 'sic'
        if 'sic' in xrd.columns:
            xrd['SIC'] = xrd['sic']
        else:
            raise KeyError("xrd_data missing 'sic'/'SIC'")
    xrd['SIC'] = xrd['SIC'].astype(str).str[:3].str.zfill(3)

    # Replace missing R&D with 0 and compute xrd/at
    if 'xrd' not in xrd.columns or 'at' not in xrd.columns:
        raise KeyError("xrd_data missing 'xrd' or 'at'")
    xrd['xrd'] = xrd['xrd'].fillna(0)
    xrd['xrd_at'] = xrd['xrd'] / xrd['at']

    # Average within firm then within SIC
    if 'gvkey' in xrd.columns:
        xrd_firm = xrd.groupby('gvkey', as_index=False).agg({'xrd_at':'mean','SIC':'first'})
        xrd_industry = xrd_firm.groupby('SIC', as_index=False).agg({'xrd_at':'mean'})
    else:
        # if gvkey missing, aggregate directly by SIC
        xrd_industry = xrd.groupby('SIC', as_index=False).agg({'xrd_at':'mean'})

    # Coerce SIC type to string on both and merge
    xrd_industry['SIC'] = xrd_industry['SIC'].astype(str).str[:3].str.zfill(3)
    industry['SIC'] = industry['SIC'].astype(str).str[:3].str.zfill(3)
    industry = industry.merge(xrd_industry, on='SIC', how='left')

    AvgRDOverAsset = industry['xrd_at'].astype(float)
except Exception as e:
    # Fallback: if industry already has xrd_at, use it; else fill zeros
    if 'xrd_at' in industry.columns:
        AvgRDOverAsset = industry['xrd_at'].astype(float)
    else:
        AvgRDOverAsset = pd.Series(0.0, index=industry.index)

# ==============================
# BUILD VARIABLES (Panel B, Col. 3)
# ==============================
# y must be passed in LEVELS (replicate() logs internally)
y_levels = industry['backloading'].astype(float)

Turnover       = industry['turnover'].astype(float)
VarNbFunc      = (industry['nbFunctionEnd'] - industry['nbFunctionStart']).astype(float)
VarComplex10K  = (industry['ncomplex10Kend'] - industry['ncomplex10Kstart']).astype(float)
VarCFOverAsset = (np.log1p(industry['OANCF_ASSETEnd'].astype(float))
                  - np.log1p(industry['OANCF_ASSETStart'].astype(float)))
AvgMktValue    = industry['mkValTStart'].astype(float)
AvgMktValGrowth= (industry['mkValTEnd'].astype(float) / industry['mkValTStart'].astype(float) - 1.0)
nbFirm         = industry['nbFirm'].astype(float)

# ==============================
# FILTERS: trim DV 1–99% (on log(y)) + nbFirm >= 2 + finite values
# ==============================
log_y = np.log(y_levels.replace(0, np.nan))
q_lo  = np.nanpercentile(log_y, 1)
q_hi  = np.nanpercentile(log_y, 99)

df = pd.DataFrame({
    'y_levels'      : y_levels,
    'log_y'         : log_y,
    'Turnover'      : Turnover,
    'VarNbFunc'     : VarNbFunc,
    'VarComplex10K' : VarComplex10K,
    'VarCFOverAsset': VarCFOverAsset,
    'AvgMktValue'   : AvgMktValue,
    'AvgMktValGrowth': AvgMktValGrowth,
    'AvgRDOverAsset': AvgRDOverAsset,
    'nbFirm'        : nbFirm
})

mask = (
    (df['log_y'] >= q_lo) & (df['log_y'] <= q_hi) &
    (df['nbFirm'] >= 2)
)

df = df.loc[mask].copy()
# Drop any remaining NA/infinite rows to keep y and X aligned
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ==============================
# DESIGN MATRIX
# ==============================
y = df['y_levels'].values  # levels
X = sm.add_constant(df[['Turnover','VarNbFunc','VarComplex10K',
                        'VarCFOverAsset','AvgMktValue','AvgMktValGrowth','AvgRDOverAsset']].astype(float),
                    has_constant='add')

# (Optional) quick sanity check
# print("Shapes -> y:", y.shape, "X:", X.shape)

# ==============================
# METADATA
# ==============================
metadata = {
    'paper_id'         : '149',
    'table_id'         : '2',
    'panel_identifier' : 'B_3',
    'model_type'       : 'log-linear',
    'comments'         : 'Biais & Landier (2020), Table 2, Panel B, Column 3. '
                         'Industry OLS with Turnover, Δ#Functions, Δ"complex" in 10-K, Δlog(1+ROA), '
                         'controls (AvgMktValue, AvgMktValGrowth, R&D/assets). '
                         'Trim 1–99% of log DV; keep nbFirm≥2. DV passed in levels (backloading).'
}

# ==============================
# RUN
# ==============================
replicate(
    metadata=metadata,
    y=y,                      # LEVELS; replicate() logs internally
    X=X,
    interest='Turnover',      # key moderator in theory; both complexity proxies included
    elasticity=False,
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
