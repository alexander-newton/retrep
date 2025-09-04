# issues
dominant_firm variable contains only 0s (no 1s), preventing replication of Table 3.

Impact: Stata code requires dominant_firm==1 filter, but this yields 0 observations from 535,099 rows.
Cause: Missing data preparation step that creates the dominant firm indicator.
Fix needed: Either the code that generates dominant_firm variable or an intermediate dataset with it properly coded
