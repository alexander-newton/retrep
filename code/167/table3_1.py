import yaml, os, pandas as pd, numpy as np, sys, statsmodels.api as sm
# make replicate() importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Load config and data (folder 167)
# -------------------------------
with open('./config.yaml') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']; INPUT_DATA_DIR = config['intermediatedata']
data_path = os.path.join(INPUT_DATA_DIR, '167/analysisdataset.dta')
df = pd.read_stata(data_path)

# -------------------------------
# Sample: 2006–2010 and non-missing G (no-spillover spec)
# -------------------------------
df = df[(df['year'].between(2006, 2010)) & df['G'].notna()].copy()

# -------------------------------
# DID interactions: Treated × {2009, 2010} using G (no spillovers)
# -------------------------------
df['treated_2009'] = ((df['G'] == 1) & (df['year'] == 2009)).astype(int)
df['treated_2010'] = ((df['G'] == 1) & (df['year'] == 2010)).astype(int)

# -------------------------------
# Dependent variable: use logodds directly (do NOT exponentiate)
# -------------------------------
df = df.dropna(subset=['logodds']).copy()
y = np.exp(df['logodds'].astype(float))

# -------------------------------
# Fixed effects: municipality + year (2008 baseline via explicit categories)
# -------------------------------
df['muni_fe'] = pd.Categorical(df['muni_ID']).codes
df['year_fe'] = pd.Categorical(df['year'], categories=[2006, 2007, 2008, 2009, 2010]).codes

# -------------------------------
# Assemble X: only DID terms + FE (no controls for col (1))
# -------------------------------
X_vars = ['treated_2009','treated_2010','muni_fe','year_fe']
X = sm.add_constant(df[X_vars], prepend=False)
fe_indices = [X.columns.get_loc('muni_fe'), X.columns.get_loc('year_fe')]

# -------------------------------
# Metadata
# -------------------------------
metadata = {
    'paper_id': '167',
    'table_id': '3',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 3, Column 1: DID without spillovers and without controls; muni & year FE (2008 baseline).'
}

# -------------------------------
# Run replication (cluster SE at municipality level)
# -------------------------------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=['treated_2009'],
    fe=fe_indices,
    elasticity=False,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': df['muni_ID']}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
