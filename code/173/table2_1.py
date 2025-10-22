import yaml, os, sys, numpy as np, pandas as pd, statsmodels.api as sm

# Pathing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ---- Load config
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ---- Load data (folder 173)
filepath = os.path.join(INPUT_DATA_DIR, '173/PovertyTraps_analysis.dta')
df = pd.read_stata(filepath)

# ---- Sanity check: show we have the columns we need
needed = ['stup','survey_wave','treat','k0','Pcows_no','Lk1','deltaLk3']
missing = [c for c in needed if c not in df.columns]
if missing:
    print("Missing expected columns:", missing)
    print("Available columns sample:", df.columns.tolist()[:40])
    # If your file uses different names, map them here before continuing.
    # raise KeyError(missing)

# ---- Sample restrictions: ultra-poor baseline (match .do)
df = df[(df['stup'] == 1) & (df['survey_wave'] == 1)].copy()

# ---- Recreate hh and filter (match .do)
df['hh'] = np.log(df['k0'] + (df['Pcows_no'] / 1000.0) + 1.0)
df = df[df['hh'] <= 3].copy()

# ---- Define threshold indicator for treatment only (Table 2 col 1)
# aboveT = 1(Lk1 >= 2.333), treat villages only
df = df[df['treat'] == 1].copy()
df = df[pd.notnull(df['Lk1']) & pd.notnull(df['deltaLk3'])].copy()
df['aboveT'] = (df['Lk1'] >= 2.333).astype(int)

# ---- y: replicate() logs internally. We want ΔlogK, so pass exp(ΔlogK) = K3/K1.
y = np.exp(df['deltaLk3'].astype(float))

# ---- X: just the threshold dummy + constant, as in: reg deltaLk3 aboveT if treat==1
X = pd.DataFrame({'aboveT': df['aboveT'].astype(int)})
X = sm.add_constant(X, prepend=False)


metadata = {
    'paper_id': '173',
    'table_id': '2',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': (
        'Balboni et al. Table 2, Col (1). DV is Δlog K (2007–2011). '
        'replicate() logs y internally; we pass y=exp(deltaLk3). '
        'Sample: stup==1, survey_wave==1, treat==1, hh<=3. Spec: OLS of ΔlogK on 1(Lk1≥2.333).'
    )
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='aboveT',
    fe=None,
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)

