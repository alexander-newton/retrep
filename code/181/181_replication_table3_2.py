import numpy as np
import os
import sys
import json
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from replication import replicate
from utils.json_utils import ReplicationJSONBuilder, load_config

config = load_config()
output_folder = config.get('output', './output')

# =====================================
# Paper 181 - Table 3, Column 2
# Same as Col 1 but at SITC 3-digit level
# =====================================

base_path = '/Users/ellenmunroe/Downloads/112606-V1/MS_AER-2010-0514-R3_Replication_Files'

# Step 1: World export shares at sitc3 level
wexp = pd.read_stata(f'{base_path}/World_exports_by_industry_annual_final.dta')
wexp = wexp.dropna(subset=['sitc4', 'isocode', 'year'])
wexp = wexp[wexp['isocode'] != 'WORLD']
wexp['sitc2'] = wexp['sitc4'].str[:2]
wexp['sitc_ind'] = wexp['sitc4'].str[:3]

# Drop aggregate misc categories
wexp = wexp[~wexp["sitc_ind"].str.match(r"..A")]
wexp = wexp[~wexp["sitc_ind"].str.match(r"..X")]
for i in range(1, 10):
    wexp = wexp[wexp['sitc2'] != f'{i}0']

wc = wexp.groupby(['isocode', 'year', 'sitc_ind'])['exportsWORLD'].sum().reset_index()
te = wc.groupby(['year', 'sitc_ind'])['exportsWORLD'].sum().reset_index().rename(columns={'exportsWORLD': 'total_exportsWORLD'})
wc = wc.merge(te, on=['year', 'sitc_ind'])
wc['world_export_share'] = wc['exportsWORLD'] / wc['total_exportsWORLD']

ae = wc.groupby(['isocode', 'year'])['exportsWORLD'].sum().reset_index().rename(columns={'exportsWORLD': 'agg_exp'})
ta = wc.groupby('year')['exportsWORLD'].sum().reset_index().rename(columns={'exportsWORLD': 'tot_agg'})
ae = ae.merge(ta, on='year')
ae['agg_share'] = ae['agg_exp'] / ae['tot_agg']
wc = wc.merge(ae[['isocode', 'year', 'agg_share']], on=['isocode', 'year'])

ue = wc[wc['isocode'] == 'USA'][['year', 'sitc_ind', 'world_export_share', 'agg_share']].copy()
ue.columns = ['year', 'sitc_ind', 'US_wes', 'US_as']
we = wc[wc['isocode'] != 'USA']

# Step 2: Imports at sitc3 level
imp = pd.read_stata(f'{base_path}/Imports_by_industry_annual_final.dta')
imp = imp.dropna(subset=['sitc4', 'isocode', 'year'])
imp['sitc2'] = imp['sitc4'].str[:2]
imp['sitc_ind'] = imp['sitc4'].str[:3]

imp = imp[~imp["sitc_ind"].str.match(r"..A")]
imp = imp[~imp["sitc_ind"].str.match(r"..X")]
for i in range(1, 10):
    imp = imp[imp['sitc2'] != f'{i}0']

mc = [c for c in ['USinfluence', 'RUSinfluence', 'cgv_democracy', 'ln_per_capita_income', 'total_gdp', 'year_in_office'] if c in imp.columns]
ia = imp.groupby(['isocode', 'year', 'sitc_ind']).agg(
    {**{'importsUSA': 'sum', 'importsWORLD': 'sum'}, **{c: 'mean' for c in mc}}
).reset_index()
ia = ia[ia['isocode'] != 'WORLD']

# Step 3: Merge and RCA
ia = ia.merge(we[['isocode', 'year', 'sitc_ind', 'world_export_share', 'agg_share']], on=['isocode', 'year', 'sitc_ind'], how='left')
ia = ia.merge(ue, on=['year', 'sitc_ind'], how='left')

ia['US_RCA'] = ia['US_wes'] / ia['US_as']
ia['importer_RCA'] = ia['world_export_share'] / ia['agg_share']
ia = ia.dropna(subset=['importsUSA', 'importsWORLD'])

for v in ['US_RCA', 'importer_RCA']:
    mn = ia[v].min(); mx = ia[v].max()
    ia[v] = (ia[v] - mn) / mx

ia = ia[~ia['isocode'].isin(['USA', 'RUS'])]

ia['USinfluence_x_US_RCA'] = ia['USinfluence'] * ia['US_RCA']
ia['USinfluence_x_importer_RCA'] = ia['USinfluence'] * ia['importer_RCA']
ia['new_leader_dum'] = np.where(ia['year_in_office'] == 1, 1,
                       np.where(ia['year_in_office'].notna() & (ia['year_in_office'] > 1), 0, np.nan))

ia = ia[(ia['year'] >= 1947) & (ia['year'] <= 1989)]
ct = ia.groupby('isocode')['importsWORLD'].transform('count')
ia = ia[ct > 0]

# Step 4: B&B correction terms
bb = pd.read_stata(f'{base_path}/Baier_Bergstrand_MR_approx_terms_final.dta')
ia['importer_isocode'] = ia['isocode']
ia['exporter_isocode'] = 'USA'
ia = ia.merge(bb, on=['importer_isocode', 'exporter_isocode', 'year'], how='left', suffixes=('', '_bb'))
ia['x1'] = ia['ln_distance'] - ia['distance_correction']
ia['x2'] = ia['contiguous'] - ia['border_correction']
ia['x3'] = ia['same_language'] - ia['language_correction']
ia['x4'] = ia['gatt'] - ia['gatt_correction']
ia['x5'] = ia['rta'] - ia['rta_correction']

# Step 5: DV
ia['importsUSA_ratio'] = ia['importsUSA'] / ia['total_gdp']
ia['ln_importsUSA'] = np.log(ia['importsUSA_ratio'])
ia = ia[np.isfinite(ia['ln_importsUSA']) & (ia['importsUSA_ratio'] > 0)]

controls = ['USinfluence', 'USinfluence_x_US_RCA', 'USinfluence_x_importer_RCA', 'US_RCA', 'importer_RCA',
            'RUSinfluence', 'ln_per_capita_income', 'new_leader_dum', 'year_in_office', 'cgv_democracy',
            'x1', 'x2', 'x3', 'x4', 'x5']
ia['year_int'] = ia['year'].astype(int)
fe_cols = ['isocode', 'year_int', 'sitc_ind']

reg_df = ia[['importsUSA_ratio'] + controls + fe_cols].dropna().reset_index(drop=True)

metadata = {
    'paper_id': '181',
    'table_id': 'Table3',
    'panel_identifier': 'Col2',
    'model_type': 'log-linear'
}

y = pd.DataFrame(reg_df['importsUSA_ratio'].values, columns=['imports_usa_gdp'])
X = reg_df[controls + fe_cols].reset_index(drop=True)

out_dir = os.path.join(output_folder, '181', 'Table3_Col2')
os.makedirs(out_dir, exist_ok=True)
y.to_parquet(os.path.join(out_dir, 'y.parquet'), index=False)
X.to_parquet(os.path.join(out_dir, 'X.parquet'), index=False)
with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

replicate(
    metadata=metadata,
    y=y.values,
    X=X[controls].values,
    interest='USinfluence',
    endog_x=None,
    z=None,
    fe=X[fe_cols].values,
    elasticity=True,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols=None,
    kwargs_ppml=None,
    fit_full_model=False,
    output=True,
    output_dir=output_folder
)
