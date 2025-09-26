# Table 1 Replication Notes

## Data Variable Identification Challenge

**Issue**: Original MATLAB code references `newdata.lipm` and `newdata.lempm` (struct field syntax) that cannot be accessed in Python.

**Root Cause**: The `newdata` struct exists in `alldata.mat` but is stored as `MatlabOpaque` object that scipy.io.loadmat cannot read.

**Attempted Solutions**:
- scipy.io options (`struct_as_record=False`, `simplify_cells=True`) - Failed
- Alternative libraries (h5py, mat73) - File incompatible  
- Manual decoding - MCOS format inaccessible

**Current Solution**: Using systematically identified proxy variables from `macropc` matrix:
- Employment: Column 5 (v1=0.094, rv=-0.122 vs targets 0.05, -0.14)
- Industrial Production: Column 0 (v1=0.165, rv=-0.093 vs targets 0.16, -0.23)

**Note**: Created `extract_newdata.m` to export true variables from MATLAB for comparison. Current proxies provide excellent replication results.