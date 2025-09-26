# Table 7 & 8 Replication Notes

## Table 7 - Education Returns by Year (1993-2000)

**Column 2 (OLS)**: Uses pre-existing `edyrsyr*` variables from dataset - gives exact match to original paper coefficients.

**Column 3 (FE)**: Required manual construction of education-year interactions (`edyrs × year_dummy`) because individual fixed effects absorbed most variation in the pre-existing `edyrsyr*` variables, making them ineffective for FE regression.


