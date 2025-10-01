# Paper 146 - Table 4, Column 1

## Memory Issue
Script crashes due to massive design matrix

### Problem
- `event_id` has 25,215 unique values → 25,214 dummy variables
- Total dummy variables: 25,665
- Design matrix: 851,864 × 25,669 = 21.9 billion elements

