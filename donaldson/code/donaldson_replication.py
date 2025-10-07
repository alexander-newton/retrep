import pandas as pd
import numpy as np
from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import statsmodels.api as sm
import tabulate


data = pd.read_csv('./donaldson/data/Dave_Donaldson_Railroads_tbl4.csv')

# REPLICATE OLS RESULTS
X = data[['RAIL', 'distid', 'year']]
y = data['realincome']

mod = DoublyRobustElasticityEstimator(endog=y, exog=X, fe=['distid', 'year'], interest='RAIL', estimator_type='nn')
ols_res = mod._fit_base_ols(weights=None, cov_type='HC3')
print(tabulate.tabulate(ols_res.summary().tables[1], headers='firstrow', tablefmt='github'))

# DISTRIBUTION OF RESIDUALS

residuals = ols_res.resid
resid_rail = residuals[data['RAIL'] == 1]
resid_no_rail = residuals[data['RAIL'] == 0]

sns.histplot(x=np.exp(residuals), hue=data['RAIL'], stat='probability')
plt.legend(title='RAIL', labels=['1: mean=1.06', '0: mean= 1.04'])
print("OLS residual means:", np.exp(resid_rail).mean(), np.exp(resid_no_rail).mean())
plt.title('Distribution of exponentiated OLS Residuals by RAIL Status')
plt.savefig('./donaldson/outputs/ols_residuals.png', transparent=True)

corr = np.exp(resid_rail).mean()/np.exp(resid_no_rail).mean()

manning_estimate = np.exp(ols_res.params[0]) * corr - 1


# PPML
X = data[['RAIL', 'distid', 'year']]
y = data['realincome']
dummies = pd.get_dummies(data[['distid', 'year']].astype('category'), drop_first=True, dtype='float')
X_resid = sm.add_constant(pd.concat([data['RAIL'], dummies], axis=1), prepend=False)
ppml = sm.GLM(y, X_resid, family=sm.families.Poisson()).fit()
ppml.summary()

rail_summary = [ppml.params['RAIL'], ppml.bse['RAIL'], ppml.pvalues['RAIL'], ppml.conf_int().loc['RAIL', 0], ppml.conf_int().loc['RAIL', 1]]

# write down rail_summary but rounded to 2 decimal places

rail_summary = [round(x, 2) for x in rail_summary]
ppml_df = pd.DataFrame([rail_summary], columns=['Estimate', 'SE', 'p-value', '95% CI Lower', '95% CI Upper']).to_markdown()

print("PPML estimate (unadjusted):", round(np.exp(rail_summary[0]) - 1,2), "SE:", rail_summary[1], "p-value:", rail_summary[2], "95% CI:", (np.exp(rail_summary[3]) - 1, np.exp(rail_summary[4]) - 1))


def estimate(data):
    X = data[['RAIL', 'distid', 'year']]
    y = data['realincome']

    mod = DoublyRobustElasticityEstimator(endog=y, exog=X, fe=['distid','year'], interest='RAIL', estimator_type='nn')
    ols_res = mod._fit_base_ols(weights=None, cov_type='HC3')

    residuals = ols_res.resid
    resid_rail = residuals[data['RAIL'] == 1]
    resid_no_rail = residuals[data['RAIL'] == 0]

    correction = np.exp(resid_rail).mean()/np.exp(resid_no_rail).mean()

    return np.exp(ols_res.params[0]) * correction - 1, correction


# implement bootstrap to get standard error of manning estimate

n_bootstraps = 1000
boot_estimates = []
corr_estimates = []
for i in range(n_bootstraps):
    boot_data = data.sample(frac=1, replace=True)
    est, corr = estimate(boot_data)
    boot_estimates.append(est)
    corr_estimates.append(corr)

boot_estimates = np.array(boot_estimates)
corr_estimates = np.array(corr_estimates)

manning_se  = np.mean((boot_estimates - manning_estimate)**2)**0.5
corr_se = np.mean((corr_estimates - corr)**2)**0.5

# Make a df with CI and print it as a markdown table no t stat or pvalue

manning_df = pd.DataFrame([[round(manning_estimate,4), round(manning_se,4), round(manning_estimate - 1.96*manning_se,4), round(manning_estimate + 1.96*manning_se,4)]], columns=['Estimate', 'SE', '95% CI Lower', '95% CI Upper']).to_markdown()

corr_df = pd.DataFrame([[round(corr,2), round(corr_se,2), round(corr - 1.96*corr_se,2), round(corr + 1.96*corr_se,2)]], columns=['Correction Factor', 'SE', '95% CI Lower', '95% CI Upper']).to_markdown()

print(manning_df)
print(corr_df)