import yaml, os, sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

# import replicate()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# --- config & data ---
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
INPUT_DATA_DIR = config['intermediatedata']
OUTPUT_DIR = config['outputdata']
df = pd.read_stata(os.path.join(INPUT_DATA_DIR, '158/wsdata_cleaned_signalling.dta'))

# --- DV: IHS(weekly earnings, unconditional) ---
dv_ihs = 'el_emp_earn_all_uncd_i'

# treatments and FE (exactly as in paper)
treat_vars = ['public','private','placebo']
interest   = 'public'
fe_cat     = 'bl_block'  # randomization-block FE

# baseline covariates (same set as Col 1/2)
covs = [
    'bl_score_num','bl_score_lit','bl_score_cft','bl_score_grit',
    'bl_score_flex','bl_score_control','bl_score_focus','bl_score_plan',
    ('el_selfest_med' if 'el_selfest_med' in df.columns else 'bl_belest_index'),
    'Edn_Matric','Edn_Cert','Edn_Degree',
    'bl_age','bl_male',
    'bl_risk_high','bl_time_med','bl_time_presentbias',
    'bl_emp_7d'
]

# --- strict checks & assemble ---
required = [dv_ihs] + treat_vars + [fe_cat] + covs
missing  = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns for Table 1 Col(3): {missing}")

# drop any pre-dummied block_* to avoid FE conflicts
block_dummy_cols = [c for c in df.columns if c.startswith('block_')]
keep_cols = [c for c in required if c not in block_dummy_cols]
d = df[keep_cols].dropna().copy()

# X build order:
X = d[treat_vars + covs].copy()
X['bl_block_fe'] = pd.Categorical(d[fe_cat]).codes
fe_indices = [X.columns.get_loc('bl_block_fe')]   # FE index BEFORE adding constant
X = sm.add_constant(X, prepend=False)

# y
y = d[dv_ihs].astype(float)

# --- metadata & run ---
metadata = {
    'paper_id': '158',
    'table_id': '1',
    'panel_identifier': '3',
    'model_type': 'IHS-linear',
    'comments': 'Table 1, Col (3): IHS(weekly earnings, unconditional) on public/private/placebo + baseline covariates + bl_block FE.'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=interest,
    fe=fe_indices,
    elasticity=False,   # DV already IHS (θ=1), no additional scaling
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)

