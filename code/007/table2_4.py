import os, yaml, numpy as np, pandas as pd, statsmodels.api as sm
from replication import replicate

# Config
with open('./config.yaml', 'r') as f:
   config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

P007 = os.path.join(INPUT_DATA_DIR, '007')
PATH_IMPORT  = os.path.join(P007, '1_all_data.dta')
PATH_WDI     = os.path.join(P007, 'wdi.dta')
PATH_WEIGHTS = os.path.join(P007, 'value_by_commodity_1.dta')

# Load data
d = pd.read_stata(PATH_IMPORT)
wdi = pd.read_stata(PATH_WDI)[['country','year','FP_WPI_TOTL','PA_NUS_ATLS']].rename(
   columns={'FP_WPI_TOTL':'ppi','PA_NUS_ATLS':'ner'}
)

# Create dyads
d['c1'] = np.minimum(d['reporteriso'].values, d['partneriso'].values)
d['c2'] = np.maximum(d['reporteriso'].values, d['partneriso'].values)
d['dyad_unord'] = d['c1'].astype(str) + '_' + d['c2'].astype(str)

# Separate import/export flows and calculate ToT levels
imports = d[d['tradeflow'] == 1][['reporteriso','partneriso','year','dyad_unord','uvi']].rename(columns={'uvi':'uvi_imp'})
exports = d[d['tradeflow'] == 2][['reporteriso','partneriso','year','uvi']].rename(columns={'uvi':'uvi_exp'})
df = imports.merge(exports, on=['reporteriso','partneriso','year'], how='outer')
df['tot_level'] = df['uvi_exp'] / df['uvi_imp']

# Calculate ToT growth factor
df.sort_values(['dyad_unord','year'], inplace=True)
same = df['dyad_unord'].eq(df['dyad_unord'].shift(1))
df['tot_growth_factor'] = np.where(same, df['tot_level']/df['tot_level'].shift(1), np.nan)

# Fixed effects
df['YearFactor'] = df['year'].astype('category')
df['DyadFactor'] = df['dyad_unord'].astype('category')

# Merge WDI
w_c1 = wdi.rename(columns={'country':'c1','ppi':'ppi_c1','ner':'ner_c1'})
w_c2 = wdi.rename(columns={'country':'c2','ppi':'ppi_c2','ner':'ner_c2'})
dw = df.merge(w_c1, on=['c1','year'], how='left').merge(w_c2, on=['c2','year'], how='left')

# Calculate ER and PPI growth factors
dw['ner_ratio'] = dw['ner_c2'] / dw['ner_c1']
dw['ppi_ratio'] = (dw['ppi_c1'] / dw['ppi_c2']) * dw['ner_ratio']

dw.sort_values(['dyad_unord','year'], inplace=True)
same = dw['dyad_unord'].eq(dw['dyad_unord'].shift(1))
dw['ner_growth'] = np.where(same, dw['ner_ratio']/dw['ner_ratio'].shift(1), np.nan)
dw['ppi_growth'] = np.where(same, dw['ppi_ratio']/dw['ppi_ratio'].shift(1), np.nan)

# Lags 0-2
for k in range(3):
   dw[f'ner_growth_L{k}'] = dw.groupby('dyad_unord')['ner_growth'].shift(k)
   dw[f'ppi_growth_L{k}'] = dw.groupby('dyad_unord')['ppi_growth'].shift(k)

# Trade weights
try:
   wgt = pd.read_stata(PATH_WEIGHTS)
   wgt = wgt.groupby(['reporteriso','partneriso']).first().reset_index()
   wgt['trade_weight'] = wgt.filter(like='share').sum(axis=1)
   dw = dw.merge(wgt[['reporteriso','partneriso','trade_weight']], on=['reporteriso','partneriso'], how='left')
   weights = dw['trade_weight']
except:
   weights = None

# Model setup
xcols = [f'ner_growth_L{k}' for k in range(3)] + [f'ppi_growth_L{k}' for k in range(3)]
model = dw[['tot_growth_factor','YearFactor','DyadFactor'] + xcols].dropna()

y = model['tot_growth_factor']
X = sm.add_constant(model[xcols], prepend=False)
clusters = model['DyadFactor']

# Replicate
replicate(
   metadata={'paper_id':'007', 'table_id':'2', 'panel_identifier':'col4', 'model_type':'log-log'},
   y=y,
   X=X,
   interest='ner_growth_L0',
   elasticity=True,
   fe=['YearFactor','DyadFactor'],
   kwargs_ols={'cov_type':'cluster','cov_kwds':{'groups':clusters}},
   weights=weights[model.index] if weights is not None else None,

)