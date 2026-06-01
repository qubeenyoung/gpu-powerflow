function summary = run_matpower_case(case_name, output_path)
%RUN_MATPOWER_CASE Run a MATPOWER power-flow case.
%
% Usage:
%   run_matpower_case
%   run_matpower_case('case9')
%   run_matpower_case('case118', '/tmp/case118-summary.json')

if nargin < 1 || isempty(case_name)
    case_name = getenv('MATPOWER_CASE');
    if isempty(case_name)
        case_name = 'case9';
    end
end

if nargin < 2
    output_path = getenv('MATPOWER_RESULT_JSON');
end

matpower_home = getenv('MATPOWER_HOME');
if isempty(matpower_home)
    matpower_home = '/opt/matpower';
end

if exist(matpower_home, 'dir') == 7
    addpath(genpath(matpower_home));
end

if exist('runpf', 'file') ~= 2
    error('MATPOWER:NotFound', ...
        'MATPOWER runpf was not found. Set MATPOWER_HOME or add MATPOWER to the MATLAB path.');
end

mpver;

pf_alg = getenv_with_default('MATPOWER_ALG', 'NR');
pf_tol = getenv_float_with_default('MATPOWER_TOL', 1e-8);
pf_max_it = getenv_float_with_default('MATPOWER_MAX_IT', 10);
pf_lin_solver = normalize_lin_solver(getenv_with_default('MATPOWER_LIN_SOLVER', 'DEFAULT'));

mpopt = mpoption( ...
    'model', 'AC', ...
    'pf.alg', pf_alg, ...
    'pf.tol', pf_tol, ...
    'pf.nr.max_it', pf_max_it, ...
    'pf.nr.lin_solver', pf_lin_solver, ...
    'verbose', 1, ...
    'out.all', 0);
results = runpf(case_name, mpopt);

summary = struct( ...
    'case', case_name, ...
    'success', logical(results.success), ...
    'baseMVA', results.baseMVA, ...
    'bus_count', size(results.bus, 1), ...
    'gen_count', size(results.gen, 1), ...
    'branch_count', size(results.branch, 1), ...
    'elapsed_seconds', results.et ...
);

fprintf('MATPOWER case: %s\n', summary.case);
fprintf('success: %d\n', summary.success);
fprintf('buses: %d, generators: %d, branches: %d\n', ...
    summary.bus_count, summary.gen_count, summary.branch_count);
fprintf('elapsed_seconds: %.6f\n', summary.elapsed_seconds);

if ~summary.success
    error('MATPOWER:PowerFlowFailed', 'MATPOWER power flow did not converge for %s.', case_name);
end

if ~isempty(output_path)
    fid = fopen(output_path, 'w');
    if fid < 0
        error('MATPOWER:OutputOpenFailed', 'Could not open output file: %s', output_path);
    end
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(summary, 'PrettyPrint', true));
    fwrite(fid, newline);
end
end

function value = getenv_with_default(name, default_value)
value = getenv(name);
if isempty(value)
    value = default_value;
end
end

function value = getenv_float_with_default(name, default_value)
raw = getenv(name);
if isempty(raw)
    value = default_value;
else
    value = str2double(raw);
    if isnan(value)
        error('MATPOWER:InvalidEnvNumber', '%s must be numeric, got: %s', name, raw);
    end
end
end

function solver = normalize_lin_solver(label)
switch upper(strtrim(label))
    case {'', 'DEFAULT'}
        solver = '';
    case {'BACKSLASH', '\'}
        solver = '\';
    otherwise
        solver = strtrim(label);
end
end
