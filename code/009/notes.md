can't find the issue!
stata: reg c3diffln_gvapc espionage patents diffln_gvapc yd_* br_* [aw=weight_workers], cluster(branch)
boottest espionage, bootcluster(branch)
I included year and branch fixed effects as well as the individual dummy variables; both approaches produced the same results, which still don’t match the original paper.