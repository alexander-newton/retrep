import yaml, os, pandas as pd, numpy as np, sys, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load config and data (folder 167)
with open('./config.yaml') as f: config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']; INPUT_DATA_DIR = config['intermediatedata']
data_path = os.path.join(INPUT_DATA_DIR, '167/analysisdataset.dta')
df = pd.read_stata(data_path)

# Restrict sample
df = df[(df['year'] >= 2006) & (df['year'] <= 2010)].copy()
df = df[df['GNT_70'].notna()].copy()

# Create treatment and spillover dummies × years
df['treated_2009']   = ((df['GNT_70'] == 1) & (df['year'] == 2009)).astype(int)
df['treated_2010']   = ((df['GNT_70'] == 1) & (df['year'] == 2010)).astype(int)
df['spillover_2009'] = ((df['GNT_70'] == 2) & (df['year'] == 2009)).astype(int)
df['spillover_2010'] = ((df['GNT_70'] == 2) & (df['year'] == 2010)).astype(int)

# Dependent variable: exp(logodds) (replicate logs internally)
y = np.exp(df['logodds'].astype(float))



controls = [
    'l_rain','l_rain2','l_temperature','pa_share',
    'prices_cattle','prices_agro','l_PIB',
    'crop_area2001','n_cattle2001',
    'maize_rain_high','soybeans_rain_high','dist_port'
]
# Fixed effects
df['muni_fe'] = pd.Categorical(df['muni_ID']).codes
df['year_fe'] = pd.Categorical(df['year'], categories=[2006,2007,2008,2009,2010]).codes

# Assemble X
X_vars = ['treated_2009','treated_2010','spillover_2009','spillover_2010'] + controls + ['muni_fe','year_fe']
X = sm.add_constant(df[X_vars], prepend=False)
fe_indices = [X.columns.get_loc('muni_fe'), X.columns.get_loc('year_fe')]

# Metadata
metadata = {
    'paper_id': '167',
    'table_id': '3',
    'panel_identifier': '4',
    'model_type': 'log-linear',
    'comments': 'Assuncao et al. Table 3 Col 4: Treated×2009, Treated×2010, Spillover×2009, Spillover×2010 with full controls + muni & year FE.'
}

# Run replication
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=['treated_2009'],
    fe=fe_indices,
    elasticity=False,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': df['muni_ID']}},
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)
//didnt save it due to results being different, happened after spillover effect
 added to the model but dont know what is wrong!