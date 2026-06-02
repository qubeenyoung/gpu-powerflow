function summary = run_matpower_case(case_name, output_path, varargin)
%RUN_MATPOWER_CASE Run MATPOWER through one maintained repository entry point.
%
% This file intentionally covers two related uses:
%
%   1. Single-case smoke run from matlab/run_matpower_case.bash.
%   2. Multi-case benchmark run from python/tests/run_matpower.py.
%
% Keeping both paths here avoids hidden MATLAB code embedded in Python strings.
% The Python runner only prepares a text file with case paths and asks this
% function to write a raw CSV. Python then normalizes that raw CSV into the
% repository-wide benchmark/runs.csv schema.
%
% Examples:
%   run_matpower_case
%   run_matpower_case('case9')
%   run_matpower_case('case118', '/tmp/case118-summary.json')
%   run_matpower_case('', '', ...
%       'Mode', 'benchmark', ...
%       'CasesFile', '/tmp/cases.txt', ...
%       'OutputCsv', '/tmp/matpower_raw.csv', ...
%       'Variant', 'matpower-default', ...
%       'LinearSolver', '', ...
%       'Warmup', 1, ...
%       'Repeats', 5)

opts = parse_options(case_name, output_path, varargin{:});
setup_matpower_path(opts.matpower_home);

switch opts.mode
    case 'single'
        summary = run_single_case(opts);
    case 'benchmark'
        summary = run_benchmark_cases(opts);
    otherwise
        error('MATPOWER:InvalidMode', 'Unsupported run mode: %s', opts.mode);
end
end

function opts = parse_options(case_name, output_path, varargin)
%PARSE_OPTIONS Resolve command-line arguments, name-value options, and env vars.
%
% Bash and Python callers both pass explicit name-value options when they need
% reproducibility. Environment variables remain useful for quick interactive
% runs and for license/path configuration inside containers.

if nargin < 1 || isempty(case_name)
    case_name = getenv_with_default('MATPOWER_CASE', 'case9');
end
if nargin < 2 || isempty(output_path)
    output_path = getenv('MATPOWER_RESULT_JSON');
end

parser = inputParser;
parser.FunctionName = 'run_matpower_case';
addParameter(parser, 'Mode', getenv_with_default('MATPOWER_RUN_MODE', 'single'));
addParameter(parser, 'CasesFile', getenv('MATPOWER_CASES_FILE'));
addParameter(parser, 'OutputCsv', getenv('MATPOWER_RAW_CSV'));
addParameter(parser, 'MatpowerHome', getenv_with_default('MATPOWER_HOME', '/opt/matpower'));
addParameter(parser, 'Variant', getenv_with_default('MATPOWER_VARIANT', 'matpower-default'));
addParameter(parser, 'LinearSolver', getenv_with_default('MATPOWER_LIN_SOLVER', 'DEFAULT'));
addParameter(parser, 'Warmup', getenv_float_with_default('MATPOWER_WARMUP', 0));
addParameter(parser, 'Repeats', getenv_float_with_default('MATPOWER_REPEATS', 1));
addParameter(parser, 'Tolerance', getenv_float_with_default('MATPOWER_TOL', 1e-8));
addParameter(parser, 'MaxIter', getenv_float_with_default('MATPOWER_MAX_IT', 50));
parse(parser, varargin{:});

opts = struct();
opts.case_name = char(string(case_name));
opts.output_path = char(string(output_path));
opts.mode = lower(strtrim(char(string(parser.Results.Mode))));
opts.cases_file = char(string(parser.Results.CasesFile));
opts.output_csv = char(string(parser.Results.OutputCsv));
opts.matpower_home = char(string(parser.Results.MatpowerHome));
opts.variant = char(string(parser.Results.Variant));
opts.linear_solver = normalize_lin_solver(parser.Results.LinearSolver);
opts.warmup = max(0, round(double(parser.Results.Warmup)));
opts.repeats = max(1, round(double(parser.Results.Repeats)));
opts.tolerance = double(parser.Results.Tolerance);
opts.max_iter = max(1, round(double(parser.Results.MaxIter)));
end

function setup_matpower_path(matpower_home)
%SETUP_MATPOWER_PATH Add MATPOWER and fail early if runpf is unavailable.
%
% The Docker image installs MATPOWER under /opt/matpower. Local users may point
% MATPOWER_HOME at another checkout. genpath is deliberate here because MATPOWER
% keeps important helpers such as makeYbus, makeSbus, and ext2int under lib/.

if exist(matpower_home, 'dir') ~= 7
    error('MATPOWER:HomeNotFound', 'MATPOWER_HOME not found: %s', matpower_home);
end
addpath(genpath(matpower_home));

if exist('runpf', 'file') ~= 2
    error('MATPOWER:NotFound', ...
        'MATPOWER runpf was not found. Set MATPOWER_HOME or add MATPOWER to the MATLAB path.');
end
end

function mpopt = make_pf_options(opts, verbose)
%MAKE_PF_OPTIONS Build the MATPOWER Newton-Raphson option block.
%
% The benchmark compares MATPOWER against cuPF's AC Newton path, so model, alg,
% tolerance, and maximum iteration count are fixed from the caller. The linear
% solver is only set when the caller asks for a concrete value; the empty string
% leaves MATPOWER's own default selection untouched.

mpopt = mpoption( ...
    'model', 'AC', ...
    'pf.alg', 'NR', ...
    'pf.tol', opts.tolerance, ...
    'pf.nr.max_it', opts.max_iter, ...
    'verbose', verbose, ...
    'out.all', 0);

if ~isempty(opts.linear_solver)
    mpopt = mpoption(mpopt, 'pf.nr.lin_solver', opts.linear_solver);
end
end

function summary = run_single_case(opts)
%RUN_SINGLE_CASE Execute one case and optionally write a compact JSON summary.
%
% This is the human-facing smoke path. It prints a short summary to stdout and
% raises an error when MATPOWER reports non-convergence, which makes the bash
% wrapper useful in CI or manual container checks.

mpver;
mpopt = make_pf_options(opts, 1);
results = runpf(opts.case_name, mpopt);

summary = struct( ...
    'case', opts.case_name, ...
    'success', logical(results.success), ...
    'baseMVA', results.baseMVA, ...
    'bus_count', size(results.bus, 1), ...
    'gen_count', size(results.gen, 1), ...
    'branch_count', size(results.branch, 1), ...
    'elapsed_seconds', get_numeric_field(results, 'et', NaN), ...
    'iterations', get_numeric_field(results, 'iterations', NaN), ...
    'output_mismatch', compute_output_mismatch(results) ...
);

fprintf('MATPOWER case: %s\n', summary.case);
fprintf('success: %d\n', summary.success);
fprintf('buses: %d, generators: %d, branches: %d\n', ...
    summary.bus_count, summary.gen_count, summary.branch_count);
fprintf('elapsed_seconds: %.6f\n', summary.elapsed_seconds);
fprintf('iterations: %g\n', summary.iterations);
fprintf('output_mismatch: %.6e\n', summary.output_mismatch);

if ~summary.success
    error('MATPOWER:PowerFlowFailed', 'MATPOWER power flow did not converge for %s.', opts.case_name);
end

if ~isempty(opts.output_path)
    write_json_summary(opts.output_path, summary);
end
end

function summary = run_benchmark_cases(opts)
%RUN_BENCHMARK_CASES Write raw MATPOWER timing rows for the Python benchmark.
%
% The raw CSV intentionally records only MATLAB-owned measurements and status:
% case name/path, warmup flag, convergence, iteration count, init/solve times,
% MATPOWER's own reported solve time, and the final reduced mismatch. Python
% enriches these rows with case dimensions and the shared benchmark manifest.

if isempty(opts.cases_file)
    error('MATPOWER:CasesFileMissing', 'Benchmark mode requires the CasesFile option.');
end
if isempty(opts.output_csv)
    error('MATPOWER:OutputCsvMissing', 'Benchmark mode requires the OutputCsv option.');
end

case_paths = read_nonempty_lines(opts.cases_file);
output_dir = fileparts(opts.output_csv);
if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

fid = fopen(opts.output_csv, 'w');
if fid < 0
    error('MATPOWER:OutputOpenFailed', 'Could not open output CSV: %s', opts.output_csv);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

csv_row(fid, {'case_name','case_path','repeat_idx','warmup','success','converged', ...
    'iterations','error_message','initialize_ms','solve_ms','reported_solve_ms','output_mismatch'});

fprintf('[%s] MATPOWER benchmark cases=%d warmup=%d repeats=%d solver=%s\n', ...
    opts.variant, numel(case_paths), opts.warmup, opts.repeats, solver_label(opts.linear_solver));

for case_idx = 1:numel(case_paths)
    case_path = case_paths{case_idx};
    [~, case_name, ~] = fileparts(case_path);
    for repeat = 0:(opts.warmup + opts.repeats - 1)
        is_warmup = repeat < opts.warmup;
        repeat_idx = repeat - opts.warmup;
        if is_warmup
            repeat_idx = repeat;
        end
        write_benchmark_row(fid, opts, case_name, case_path, repeat_idx, is_warmup);
    end
end

summary = struct( ...
    'mode', 'benchmark', ...
    'variant', opts.variant, ...
    'case_count', numel(case_paths), ...
    'output_csv', opts.output_csv ...
);
end

function write_benchmark_row(fid, opts, case_name, case_path, repeat_idx, is_warmup)
%WRITE_BENCHMARK_ROW Run one initialize+solve cycle and append one CSV row.
%
% The timing split mirrors the Python/cuPF runners:
%   initialize_ms = loadcase + mpoption construction
%   solve_ms      = runpf wall time
% MATPOWER also exposes results.et, which is stored separately because it may
% exclude or include slightly different internal work depending on MATPOWER.

try
    t_init = tic;
    mpc = loadcase(case_path);
    mpopt = make_pf_options(opts, 0);
    initialize_ms = toc(t_init) * 1000.0;

    t_solve = tic;
    results = runpf(mpc, mpopt);
    solve_ms = toc(t_solve) * 1000.0;

    reported_solve_ms = get_numeric_field(results, 'et', NaN) * 1000.0;
    iterations = get_numeric_field(results, 'iterations', NaN);
    output_mismatch = compute_output_mismatch(results);

    csv_row(fid, {case_name, case_path, repeat_idx, double(is_warmup), ...
        1, double(results.success), iterations, '', initialize_ms, solve_ms, ...
        reported_solve_ms, output_mismatch});

    if ~is_warmup
        fprintf('[%s][OK] %s repeat=%d init_ms=%.3f solve_ms=%.3f resid=%.3e\n', ...
            opts.variant, case_name, repeat_idx, initialize_ms, solve_ms, output_mismatch);
    end
catch exc
    csv_row(fid, {case_name, case_path, repeat_idx, double(is_warmup), ...
        0, 0, NaN, getReport(exc, 'basic', 'hyperlinks', 'off'), NaN, NaN, NaN, NaN});
    fprintf(2, '[%s][FAIL] %s: %s\n', opts.variant, case_name, exc.message);
end
end

function mismatch = compute_output_mismatch(results)
%COMPUTE_OUTPUT_MISMATCH Rebuild MATPOWER's reduced Newton mismatch.
%
% cuPF compares against the reduced vector F = [P(pv); P(pq); Q(pq)] in
% internal bus ordering. ext2int converts the solved MATPOWER struct to that
% ordering before makeYbus/makeSbus/bustypes are evaluated.

try
    define_constants;
    int_results = ext2int(results);
    [Ybus, ~, ~] = makeYbus(int_results.baseMVA, int_results.bus, int_results.branch);
    Sbus = makeSbus(int_results.baseMVA, int_results.bus, int_results.gen);
    [~, pv, pq] = bustypes(int_results.bus, int_results.gen);
    V = int_results.bus(:, VM) .* exp(1j * pi / 180 * int_results.bus(:, VA));
    mis = V .* conj(Ybus * V) - Sbus;
    F = [real(mis(pv)); real(mis(pq)); imag(mis(pq))];
    if isempty(F)
        mismatch = 0.0;
    else
        mismatch = norm(F, inf);
    end
catch mismatch_error %#ok<NASGU>
    mismatch = NaN;
end
end

function lines = read_nonempty_lines(path)
fid = fopen(path, 'r');
if fid < 0
    error('MATPOWER:CasesFileOpenFailed', 'Could not open cases file: %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
raw = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
raw = raw{1};
lines = {};
for i = 1:numel(raw)
    value = strtrim(raw{i});
    if ~isempty(value)
        lines{end + 1} = value; %#ok<AGROW>
    end
end
end

function write_json_summary(output_path, summary)
output_dir = fileparts(output_path);
if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end
fid = fopen(output_path, 'w');
if fid < 0
    error('MATPOWER:OutputOpenFailed', 'Could not open output file: %s', output_path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, jsonencode(summary, 'PrettyPrint', true));
fwrite(fid, newline);
end

function csv_row(fid, values)
for k = 1:numel(values)
    if k > 1
        fprintf(fid, ',');
    end
    fprintf(fid, '%s', csv_cell(values{k}));
end
fprintf(fid, '\n');
end

function out = csv_cell(value)
if isnumeric(value) || islogical(value)
    if isempty(value) || any(isnan(value(:)))
        out = '';
    else
        out = sprintf('%.17g', value);
    end
else
    out = char(string(value));
end
if contains(out, '"')
    out = strrep(out, '"', '""');
end
if contains(out, ',') || contains(out, '"') || contains(out, newline)
    out = ['"' out '"'];
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
solver = char(string(label));
switch upper(strtrim(solver))
    case {'', 'DEFAULT'}
        solver = '';
    case {'BACKSLASH', '\'}
        solver = '\';
    otherwise
        solver = strtrim(solver);
end
end

function label = solver_label(solver)
if isempty(solver)
    label = 'default';
else
    label = solver;
end
end

function value = get_numeric_field(s, name, default_value)
if isfield(s, name) && isnumeric(s.(name)) && isscalar(s.(name))
    value = s.(name);
else
    value = default_value;
end
end
