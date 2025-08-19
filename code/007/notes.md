# Issues
Data file `1_all_data.dta` failed to load (memory/timeout error)

## Approach
- **Equation (20)**: Δtot_ij,t = λ_ij + δ_t + Σβ_k*Δe_ij,t-k + Σθ_k*Δppi_ij,t-k + ε_ij,t
- **Y variable**: ToT growth factor (levels) → log(y) = Δlog(ToT)  
- **X variables**: ER & PPI growth factors with lags 0-2
- **Fixed effects**: dyad + year
- **Standard errors**: Clustered at dyad level
- **Weights**: Trade-weighted for Column 4

## Result
Could not complete - data loading failed at `pd.read_stata(PATH_IMPORT)`