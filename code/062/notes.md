# issues:
1. Data Type Conflicts: Ethnicity strings in IFLS; Yes/No strings in ZFPS binary variables → required conversion/ used another variable in the dataset; eth_group 
2. Singleton Dropping: FE estimation drops single-observation groups, Stata reports original N, Python shows actual ...
3. Variable Names: Dataset names differ from Stata code (junior_plus vs junior)
5. Cluster SE: Variable length mismatch after transformations
6. only saved 4_6 with closest results and 4_11 (less satisfying!) but those count as the main results of this table.