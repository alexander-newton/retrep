# Paper 129 - Replication Notes

## Data Preparation
Uses prepared datasets created from raw data using Stata scripts. All preparation scripts are included:
- `table2_datapreparation.do`, `table5_datapreparation.do`, `table6_datapreparation.do`

## Important Setup
**Change directory at the beginning of each Stata script before running.** Update `cd` commands and verify file paths point to correct raw data locations.

## Files
- **Table 2**: `table2_A_9.py`, `table2_B_9.py`
- **Table 5**: `table5_A_2.py`, `table5_A_4.py`, `table5_B_2.py`, `table5_B_4.py`
- **Table 6**: `table6_B_1_population.py`, `table6_B_1_humancapital.py`

## Technical Notes
- Uses `replicate()` function with internal logging
- IV regressions implement Stata's `partial()` behavior

