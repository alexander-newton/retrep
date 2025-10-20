import yaml
import os
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Load configuration
# -------------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# -------------------------------
# Load data (folder 171)
# -------------------------------
data_path = os.path.join(INPUT_DATA_DIR, '171/UsingData.dta')
df = pd.read_stata(data_path)

# -------------------------------
# Sample restriction (drop if Year2 == 0)
# -------------------------------
df = df[df['Year2'] != 0].copy()

# -------------------------------
# Dependent variable in LEVELS (replicate() logs internally)
# -------------------------------
y = np.exp(df['lUnitPrice'].astype(float))

# -------------------------------
# Core vars (Table 2, Col 5)
# -------------------------------
treat_vars = ['Incentives0', 'Rules0', 'Both0']
num_controls = ['NCCs'] if 'NCCs' in df.columns else ['NCC']
fe_vars = ['NewItemID', 'strata']
qty_var = 'lQuantity'

# --- Attribute controls ---
# CATEGORICAL (i.$ivars) → dummies
ivars = [
    'BRAND_It4834','GRADE_It4834','TYPE_It4834','WITH_RUBBER_It4834','Unit_It4834',
    'Unit_It921','BRAND_It938','COUNTRY_OF_ORIGIN_It938','HANDLE_LENGTH_FT_It938',
    'HANDLE_MATERIAL_It938','WIPER_MATERIAL_It938','Unit_It938','BRAND_AND_MODEL_It1001',
    'NUMBER_OF_DIGITS_It1001','TYPE_OF_CALCULATOR_It1001','Unit_It2762','BRAND_It999',
    'SIZE_It999','Unit_It999','BRAND_AND_MODEL_It943','COUNTRY_OF_ORIGIN_It943','DIGITAL_It943',
    'FITTING_CHARGES_It943','LOCK_SIZE_It943','MATERIAL_It943','TYPE_It943','Unit_It943',
    'BRAND_It1030','COLOR_It1030','SIZE_It1030','WITH_INK_It1030','Unit_It1030',
    'MATERIAL_It936','SIZE_It936','TYPE_It936','WITH_HANDLE_It936','Unit_It936',
    'ACID_CLEANERTIZAAB_It5107','BRAND_It5107','ENVIRONMENT_FRIENDLY_It5107','MAKE_It5107',
    'SCENTED_It5107','STATE_It5107','Unit_It5107','BRAND_It1009','CLIP_A_It1009',
    'COUNTRY_OF_ORIGIN_It1009','COVER_MATERIAL_It1009','CUSTOMIZED_PRINTING_It1009',
    'FILE_TYPE_It1009','SIZE_It1009','Unit_It1009','FRAME_TYPE_It20433','MATERIAL_It20433',
    'PRINT_ON_BOTH_SIDES_It20433','WITH_ROPE_It20433','WITH_STAND_It20433','WITH_STICK_It20433',
    'MODEL_It998','SIZE_It998','COLOR_It1032','DOUBLE_SIDED_It1032','ON_GENERATOR_It1032',
    'SIZE_It1032','WITH_BINDING_It1032','PAPER_QUALITY_GSM_It1032','MODEL_It622',
    'REFILL_OR_NEW_CARTRIDGE_It622','MATERIAL_It991','PRINTED_It991','WITH_ZIP_It991','Unit_It991',
    'ANTISEPTIC_It929','BRAND_It929','STATE_It929','TYPE_OF_SOAP_It929','Unit_It929',
    'BRAND_It3906','TYPE_OF_BULB_It3906','WATT_It3906','WITH_FITTING_It3906',
    'WITH_HOLDER_OR_PATTI_It3906','Unit_It3906','BRAND_It1360','HANDLE_MATERIAL_It1360',
    'TYPE_It1360','Unit_It1360','NAME_OF_NEWSPAPER_It3346','BINDING_It994','BRAND_It994',
    'COLORED_PAGES_It994','CUSTOMIZED_PRINTING_It994','PAGE_SIZE_It994','PAGE_WEIGHT_GSM_It994',
    'TYPE_OF_REGISTER_It994','Unit_It994','BRAND_It992','COLOURED_PAGES_It992','SIZE_It992',
    'WEIGHT_PER_SHEET_It992','Unit_It992','COLOR_It989','MODEL_It989','PEN_TYPE_It989','Unit_It989',
    'TOWEL_MATERIAL_It927','TYPE_OF_TOWEL_It927','Unit_It927','MANUFACTURER_It3507',
    'MATERIAL_It3507','SIZE_It3507','TYPE_OF_PIPE_It3507','Unit_It3507',
    'NewUnit_It5107','NewUnit_It1009','NewUnit_It991','NewUnit_It4834','NewUnit_It921',
    'NewUnit_It938','NewUnit_It2762','NewUnit_It999','NewUnit_It943','NewUnit_It1030',
    'NewUnit_It936','NewUnit_It929','NewUnit_It3906','NewUnit_It1360','NewUnit_It994',
    'NewUnit_It992','NewUnit_It989','NewUnit_It927','NewUnit_It3507','NewUnit_It622',
    'NewUnit_It998','NewUnit_It1001','NewUnit_It1032','NewUnit_It3346','NewUnit_It20433',
    'sizeL','qual'
]
ivars = [v for v in ivars if v in df.columns]

# NUMERIC ($nvars) — filter to truly numeric columns only
candidate_nvars = [
    'mUnit_It5107','mUnit_It1009','NUMBER_OF_COLORS_It20433','NUMBER_OF_RINGS_It20433','AREA_It20433',
    'mNUMBER_OF_COLORS_It20433','mNUMBER_OF_RINGS_It20433','mAREA_It20433',
    'AREA_It991','mAREA_It991',
    'BarSize_It929','BottleSize_It929','PacketSize_It929','mBarSize_It929','mBottleSize_It929','mPacketSize_It929',
    'HANDLE_LENGTH_It1360','mHANDLE_LENGTH_It1360',
    'NUMBER_OF_PAGES_It994','mNUMBER_OF_PAGES_It994',
    'THICKNESS_MM_It989','mTHICKNESS_MM_It989',
    'SIZE_It927','mSIZE_It927',
    'DAIMETER_It3507','mDAIMETER_It3507',
    'WEIGHT_PER_SHEET_It992'
]
candidate_nvars = [c for c in candidate_nvars if c in df.columns]
# keep only columns that are numeric (non-numeric will be excluded)
nvars = [c for c in candidate_nvars if pd.api.types.is_numeric_dtype(df[c])]

# -------------------------------
# Interactions: NewItemID#c.lQuantity
# -------------------------------
df[qty_var] = df[qty_var].astype(float)
item_d = pd.get_dummies(df['NewItemID'], prefix='item', drop_first=True)
for c in item_d.columns:
    df[f'{c}__lq'] = item_d[c].astype(float) * df[qty_var]
interaction_terms = [c for c in df.columns if c.endswith('__lq')]

# -------------------------------
# Build X (include FE identifiers too)
# -------------------------------
X_vars = treat_vars + num_controls + nvars + [qty_var] + interaction_terms

# Add dummies for all categorical attributes (i.$ivars)
for c in ivars:
    d = pd.get_dummies(df[c], prefix=c, drop_first=True)
    if d.shape[1]:
        df = pd.concat([df, d], axis=1)
        X_vars += list(d.columns)

# Include FE id columns in X (as requested)
X_vars += fe_vars

X = df[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# -------------------------------
# One dropna pass (consistent with your example)
# -------------------------------
required_vars = X_vars + ['lUnitPrice', 'ExpInCtrl', 'CostCenterCode']
df = df.dropna(subset=required_vars)
X = X.loc[df.index]
y = y.loc[df.index]

print(f"Sample size after dropping missing values: {len(df)}")  # should be 11771

# -------------------------------
# FE as integer indices (critical fix)
# -------------------------------
fe_indices = [X.columns.get_loc('NewItemID'), X.columns.get_loc('strata')]

# -------------------------------
# Weights & clustering
# -------------------------------
weights = df['ExpInCtrl'].astype(float)       # [aweight=ExpInCtrl]
cluster = df['CostCenterCode']                # cl(CostCenter)

# -------------------------------
# Metadata
# -------------------------------
metadata = {
    'paper_id': '171',
    'table_id': '2',
    'panel_identifier': '5',
    'model_type': 'log-linear',
    'comments': 'Table 2 col (5): lUnitPrice on treatments w/ attribute controls, item×lQuantity slopes, i.NewItemID & i.strata FE; aweight=ExpInCtrl; cluster=CostCenterCode.'
}

# -------------------------------
# Run replication
# -------------------------------
replicate(
    metadata=metadata,
    y=y,                              # levels; replicate() logs internally
    X=X,
    interest=['Rules0'],
    fe=fe_indices,                    # <- pass FE as integer indices
    elasticity=False,
    weights=weights,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
