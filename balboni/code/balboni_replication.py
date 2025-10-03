from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Clean Data

df = pd.read_stata("./balboni/data/data_table_2.dta")

df_treated = df.query('treat==1')
y = df_treated['deltaLk3']
df_treated['Lk1'] = df_treated['Lk1'] - 2.333
final_x= df_treated[['aboveT', 'Lk1']]
final_x['interaction'] = df_treated['aboveT'] * df_treated['Lk1']
x = final_x[['aboveT']]


# Replicate OLS results
x = sm.add_constant(x, prepend=False)
ols_res = sm.OLS(y, x).fit()
ols_res.summary()

# Descriptive Statistics
df_treated['grat'] = np.exp(y) - 1
mean_by_threshold = df_treated.groupby('aboveT')['grat'].mean()
sd_by_threshold = df_treated.groupby('aboveT')['grat'].std()
frequency_by_threshold = df_treated.groupby('aboveT')['grat'].describe()
df_treated['growth_bins'] = pd.cut(df_treated['grat'], bins=[-1, -0.5, -0.14, 0, 0.16, 0.5, 1.2])
percentage_in_growth_bins = df_treated.groupby(['aboveT'])['growth_bins'].value_counts(normalize=True).unstack().fillna(0).transpose() * 100
print(percentage_in_growth_bins.round(2).to_markdown())


# Manning 1998 corrections
residuals = ols_res.resid
residuals_belowT = residuals[x['aboveT'] == 0]
residuals_aboveT = residuals[x['aboveT'] == 1]

manning_belowT = np.exp(residuals_belowT).mean()
manning_aboveT = np.exp(residuals_aboveT).mean()

growth_for_those_belowT = np.exp(ols_res.params['const']) * manning_belowT - 1
growth_for_those_aboveT = np.exp(ols_res.params['const'] + ols_res.params['aboveT']) * manning_aboveT - 1

semi_elasticity_estimate = np.exp(ols_res.params['aboveT']) * (manning_aboveT/manning_belowT) - 1

# OLS estimates of growth rate

ols_g_res = sm.OLS(np.exp(y) - 1, x).fit()
ols_g_res.summary()
from tabulate import tabulate


print(tabulate( ols_g_res.summary().tables[1], headers='firstrow', tablefmt='github'))



# PPML Estimates
ppml_res = sm.GLM(np.exp(y), x, family=sm.families.Poisson()).fit(cov_type='HC3')

print(tabulate(ppml_res.summary().tables[1], headers='firstrow', tablefmt='github'))

# Quantile Regression Estimates

# plot
qs = np.arange(0.05, 1, 0.05)


rows = []
for q in qs:
    res = sm.QuantReg(y, x).fit(q=q, cov_type="robust")
    ci = res.conf_int()
    p = res.params
    cov = res.cov_params()

    # intercept + dummy
    comb = p["const"] + p["aboveT"]
    var = cov.loc["const","const"] + cov.loc["aboveT","aboveT"] + 2*cov.loc["const","aboveT"]
    se = np.sqrt(var)
    lo, hi = comb - 1.96*se, comb + 1.96*se

    rows.append({
        "q": q,
        "const": p["const"], "const_lo": ci.loc["const",0], "const_hi": ci.loc["const",1],
        "dummy": p["aboveT"], "dummy_lo": ci.loc["aboveT",0], "dummy_hi": ci.loc["aboveT",1],
        "const+dummy": comb, "const+dummy_lo": lo, "const+dummy_hi": hi
    })

out = pd.DataFrame(rows)

# plot
fig, axes = plt.subplots(3, 1, figsize=(12,9), sharex=True, sharey=True)
for ax, col, label in [(axes[0], "const", r"$\beta_0$"),
                       (axes[1], "dummy", r"$\beta_1$"),
                       (axes[2], "const+dummy", r"$\beta_0 + \beta_1$")]:
    ax.plot(out["q"], out[col], marker="o")
    ax.fill_between(out["q"], out[f"{col}_lo"], out[f"{col}_hi"], alpha=0.2, step="mid")
    ax.axhline(0, linewidth=1, color="gray")
    ax.set_ylabel(label)

axes[2].set_xlabel("Quantile τ")
axes[0].set_title("Quantile Regression Estimates of Growth Rate", y=1.02, fontsize=16)
plt.tight_layout()
fig.savefig('./balboni/outputs/balboni_quantile_regression.png', dpi=300, transparent=True)










