# issues: don't match
what i did:
Loaded the experimental data: 424 saver-monitor pairs from Karnataka villages

Extracted qij signaling values: Averaged 1000 MCMC posterior draws for each pair using colMeans() in R

Standardized the qij values: Subtracted mean, divided by standard deviation

Ran log-linear regression: Savings (in levels) on standardized qij, with village clustering

Result: 1 SD increase in qij_ARD → 10.5% increase in savings (paper reports 18.5%)