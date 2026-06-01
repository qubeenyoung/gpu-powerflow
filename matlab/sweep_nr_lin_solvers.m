function results_table = sweep_nr_lin_solvers(cases, solvers, output_path)
%SWEEP_NR_LIN_SOLVERS Sweep MATPOWER AC/NR linear solvers across cases.
%
% Defaults select representative 100+ bus MATPOWER cases by size bucket.
% Usage:
%   sweep_nr_lin_solvers
%   sweep_nr_lin_solvers('case118,case300', 'DEFAULT,LU3,LU5', 'results/out.csv')

if nargin < 1 || isempty(cases)
    cases = getenv('MATPOWER_SWEEP_CASES');
end
if nargin < 2 || isempty(solvers)
    solvers = getenv('MATPOWER_LIN_SOLVERS');
end
if nargin < 3 || isempty(output_path)
    output_path = getenv('MATPOWER_SWEEP_CSV');
    if isempty(output_path)
        output_path = fullfile('results', 'matpower_ac_nr_lin_solver_sweep.csv');
    end
end

setup_matpower_path();

case_list = parse_list(cases, default_cases());
solver_labels = parse_list(solvers, default_solver_labels());

pf_tol = getenv_float_with_default('MATPOWER_TOL', 1e-8);
pf_max_it = getenv_float_with_default('MATPOWER_MAX_IT', 10);
repeat_count = getenv_float_with_default('MATPOWER_SWEEP_REPEATS', 1);
repeat_count = max(1, round(repeat_count));

fprintf('MATPOWER AC/NR linear solver sweep\n');
fprintf('cases: %d, solvers: %d, repeats: %d\n', numel(case_list), numel(solver_labels), repeat_count);
fprintf('pf.tol: %.3g, pf.nr.max_it: %d\n', pf_tol, pf_max_it);

rows = {};
for ci = 1:numel(case_list)
    case_name = case_list{ci};
    fprintf('\n[%d/%d] loading %s\n', ci, numel(case_list), case_name);
    try
        mpc = loadcase(case_name);
        bus_count = size(mpc.bus, 1);
        gen_count = size(mpc.gen, 1);
        branch_count = size(mpc.branch, 1);
        load_error_id = '';
        load_error_message = '';
    catch ME
        mpc = [];
        bus_count = NaN;
        gen_count = NaN;
        branch_count = NaN;
        load_error_id = ME.identifier;
        load_error_message = ME.message;
        fprintf('  load failed: %s\n', ME.message);
    end

    for si = 1:numel(solver_labels)
        solver_label = solver_labels{si};
        solver_value = normalize_lin_solver(solver_label);
        for ri = 1:repeat_count
            started_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
            success = false;
            iterations = NaN;
            matpower_seconds = NaN;
            wall_seconds = NaN;
            error_id = load_error_id;
            error_message = load_error_message;

            if ~isempty(mpc)
                timer = [];
                try
                    mpopt = mpoption( ...
                        'model', 'AC', ...
                        'pf.alg', 'NR', ...
                        'pf.tol', pf_tol, ...
                        'pf.nr.max_it', pf_max_it, ...
                        'pf.nr.lin_solver', solver_value, ...
                        'verbose', 0, ...
                        'out.all', 0);

                    timer = tic;
                    results = runpf(mpc, mpopt);
                    wall_seconds = toc(timer);

                    success = logical(results.success);
                    iterations = get_numeric_field(results, 'iterations', NaN);
                    matpower_seconds = get_numeric_field(results, 'et', NaN);
                    error_id = '';
                    error_message = '';
                catch ME
                    if ~isempty(timer)
                        wall_seconds = toc(timer);
                    end
                    error_id = ME.identifier;
                    error_message = ME.message;
                end
            end

            fprintf('  %-10s repeat %d/%d success=%d iter=%g wall=%.6f\n', ...
                solver_label, ri, repeat_count, success, iterations, wall_seconds);

            rows(end + 1, :) = { ...
                started_utc, case_name, bus_count, gen_count, branch_count, ...
                'AC', 'NR', pf_tol, pf_max_it, solver_label, solver_value, ...
                ri, success, iterations, matpower_seconds, wall_seconds, ...
                error_id, error_message ...
            }; %#ok<AGROW>
        end
    end
end

results_table = cell2table(rows, 'VariableNames', { ...
    'started_utc', 'case_name', 'bus_count', 'gen_count', 'branch_count', ...
    'model', 'pf_alg', 'pf_tol', 'pf_nr_max_it', 'lin_solver_label', 'lin_solver_value', ...
    'repeat', 'success', 'iterations', 'matpower_seconds', 'wall_seconds', ...
    'error_id', 'error_message' ...
});

if ~isempty(output_path)
    output_dir = fileparts(output_path);
    if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
        mkdir(output_dir);
    end
    writetable(results_table, output_path);
    fprintf('\nwrote %s\n', output_path);
end
end

function setup_matpower_path()
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
end

function values = parse_list(raw, defaults)
if isempty(raw)
    values = defaults;
    return;
end
if iscell(raw)
    values = raw;
    return;
end
if isstring(raw)
    raw = char(raw);
end
parts = regexp(raw, '[,\s]+', 'split');
values = {};
for i = 1:numel(parts)
    value = strtrim(parts{i});
    if ~isempty(value)
        values{end + 1} = value; %#ok<AGROW>
    end
end
if isempty(values)
    values = defaults;
end
end

function cases = default_cases()
cases = { ...
    'case118', ...
    'case300', ...
    'case533mt_hi', ...
    'case1354pegase', ...
    'case2869pegase', ...
    'case3375wp', ...
    'case6468rte', ...
    'case9241pegase', ...
    'case13659pegase', ...
    'case_ACTIVSg25k', ...
    'case_ACTIVSg70k' ...
};
end

function solvers = default_solver_labels()
solvers = {'DEFAULT', 'BACKSLASH', 'LU', 'LU3', 'LU4', 'LU5'};
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

function value = get_numeric_field(s, name, default_value)
if isfield(s, name) && isnumeric(s.(name)) && isscalar(s.(name))
    value = s.(name);
else
    value = default_value;
end
end
