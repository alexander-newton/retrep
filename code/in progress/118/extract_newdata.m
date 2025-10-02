% MATLAB Script to Extract newdata.lipm and newdata.lempm
% Run this in MATLAB to export the true variables to CSV

% Load the original data file
load('alldata.mat');

% Check if newdata exists and has the required fields
if exist('newdata', 'var')
    fprintf('Found newdata variable!\n');
    
    % Check available fields
    if isstruct(newdata)
        fprintf('newdata fields: ');
        disp(fieldnames(newdata));
        
        % Export lipm (Industrial Production)
        if isfield(newdata, 'lipm')
            csvwrite('../../intermediatedata/118/lipm_true.csv', newdata.lipm);
            fprintf('✅ Exported newdata.limp to intermediatedata/118/lipm_true.csv\n');
            fprintf('lipm size: [%d x %d]\n', size(newdata.lipm));
            fprintf('lipm sample: %.4f, %.4f, %.4f, %.4f, %.4f\n', newdata.lipm(1:5));
        else
            fprintf('❌ Field lipm not found in newdata\n');
        end
        
        % Export lempm (Employment)
        if isfield(newdata, 'lempm')
            csvwrite('../../intermediatedata/118/lempm_true.csv', newdata.lempm);
            fprintf('✅ Exported newdata.lempm to intermediatedata/118/lempm_true.csv\n');
            fprintf('lempm size: [%d x %d]\n', size(newdata.lempm));
            fprintf('lempm sample: %.4f, %.4f, %.4f, %.4f, %.4f\n', newdata.lempm(1:5));
        else
            fprintf('❌ Field lempm not found in newdata\n');
        end
        
    else
        fprintf('newdata exists but is not a struct\n');
        fprintf('newdata type: %s\n', class(newdata));
    end
    
else
    fprintf('❌ newdata variable not found in workspace\n');
    fprintf('Available variables:\n');
    whos
end

% Also export masterdates for reference
if exist('masterdates', 'var')
    csvwrite('../../intermediatedata/118/masterdates.csv', masterdates);
    fprintf('✅ Exported masterdates to intermediatedata/118/masterdates.csv\n');
end
