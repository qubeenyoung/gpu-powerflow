function manifest = export_exapf_numeric_cases(cases_file, output_dir, matpower_home)
%EXPORT_EXAPF_NUMERIC_CASES Export original MATPOWER cases after loadcase().
%
% ExaPF's MATLAB parser does not execute MATPOWER helper code such as idx_bus
% assignments or post-load unit conversions. This exporter keeps the original
% MATPOWER semantics by calling loadcase() first, then writes only numeric
% MATPOWER v2 fields that ExaPF can parse statically.

if nargin < 3 || isempty(matpower_home)
    matpower_home = getenv_with_default('MATPOWER_HOME', '/opt/matpower');
end
if nargin < 2 || isempty(output_dir)
    error('export_exapf_numeric_cases:MissingOutputDir', 'output_dir is required');
end
if nargin < 1 || isempty(cases_file)
    error('export_exapf_numeric_cases:MissingCasesFile', 'cases_file is required');
end

setup_matpower_path(matpower_home);

case_paths = read_nonempty_lines(cases_file);
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

manifest_path = fullfile(output_dir, 'cases.txt');
manifest_fid = fopen(manifest_path, 'w');
if manifest_fid < 0
    error('export_exapf_numeric_cases:OpenFailed', 'Could not open %s', manifest_path);
end
cleanup = onCleanup(@() fclose(manifest_fid)); %#ok<NASGU>

manifest = struct('case_count', numel(case_paths), 'output_dir', output_dir, 'cases_file', manifest_path);
for k = 1:numel(case_paths)
    case_path = case_paths{k};
    [~, case_name, ~] = fileparts(case_path);
    out_path = fullfile(output_dir, [sanitize_function_name(case_name) '_exapf_numeric.m']);
    try
        mpc = ext2int(loadcase(case_path));
        write_numeric_case(out_path, case_name, mpc);
        fprintf(manifest_fid, '%s\n', out_path);
        fprintf('[exapf-export][OK] %s -> %s\n', case_name, out_path);
    catch exc
        fprintf(2, '[exapf-export][FAIL] %s: %s\n', case_name, exc.message);
    end
end
end

function setup_matpower_path(matpower_home)
if exist(matpower_home, 'dir') ~= 7
    error('export_exapf_numeric_cases:MatpowerHomeNotFound', 'MATPOWER_HOME not found: %s', matpower_home);
end
addpath(genpath(matpower_home));
if exist('loadcase', 'file') ~= 2
    error('export_exapf_numeric_cases:MatpowerNotFound', 'MATPOWER loadcase was not found');
end
end

function write_numeric_case(out_path, case_name, mpc)
fid = fopen(out_path, 'w');
if fid < 0
    error('export_exapf_numeric_cases:OpenFailed', 'Could not open %s', out_path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

mpc = normalize_for_exapf(mpc);

function_name = sanitize_function_name([case_name '_exapf_numeric']);
fprintf(fid, 'function mpc = %s\n', function_name);
fprintf(fid, '%% Generated from original MATPOWER case via loadcase().\n');
fprintf(fid, '%% ExaPF compatibility: multiple REF buses are normalized to one REF plus PV buses.\n');
fprintf(fid, 'mpc.version = ''2'';\n');
fprintf(fid, 'mpc.baseMVA = %.17g;\n\n', double(mpc.baseMVA));
write_matrix(fid, 'bus', mpc.bus);
write_matrix(fid, 'gen', mpc.gen);
write_matrix(fid, 'branch', mpc.branch);
if isfield(mpc, 'gencost') && ~isempty(mpc.gencost)
    write_matrix(fid, 'gencost', mpc.gencost);
else
    write_matrix(fid, 'gencost', default_gencost(size(mpc.gen, 1)));
end
end

function mpc = normalize_for_exapf(mpc)
%NORMALIZE_FOR_EXAPF Keep MATPOWER numeric data, but satisfy ExaPF limitations.
%
% ExaPF's PowerNetwork constructor assumes exactly one REF bus. MATPOWER and
% cuPF support multiple REF buses, so for ExaPF-only inputs we keep the first
% REF bus and demote additional REF buses to PV. This is the same practical
% convention commonly used when importing multi-slack cases into single-slack
% solvers.

REF = 3;
PV = 2;
BUS_TYPE = 2;

ref_rows = find(mpc.bus(:, BUS_TYPE) == REF);
if numel(ref_rows) > 1
    mpc.bus(ref_rows(2:end), BUS_TYPE) = PV;
end
end

function write_matrix(fid, name, matrix)
fprintf(fid, 'mpc.%s = [\n', name);
for r = 1:size(matrix, 1)
    fprintf(fid, '    ');
    for c = 1:size(matrix, 2)
        if c > 1
            fprintf(fid, '\t');
        end
        fprintf(fid, '%s', format_number(matrix(r, c)));
    end
    fprintf(fid, ';\n');
end
fprintf(fid, '];\n\n');
end

function gencost = default_gencost(n_gen)
gencost = zeros(n_gen, 7);
gencost(:, 1) = 2;
gencost(:, 4) = 3;
end

function out = format_number(value)
value = double(value);
if isnan(value)
    out = 'NaN';
elseif isinf(value) && value > 0
    out = 'Inf';
elseif isinf(value) && value < 0
    out = '-Inf';
elseif abs(value - round(value)) < 1e-12 && abs(value) < 1e12
    out = sprintf('%d', round(value));
else
    out = sprintf('%.17g', value);
end
end

function lines = read_nonempty_lines(path)
fid = fopen(path, 'r');
if fid < 0
    error('export_exapf_numeric_cases:OpenFailed', 'Could not open %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
lines = {};
while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    line = strtrim(line);
    if ~isempty(line)
        lines{end + 1} = line; %#ok<AGROW>
    end
end
end

function value = getenv_with_default(name, default_value)
value = getenv(name);
if isempty(value)
    value = default_value;
end
end

function name = sanitize_function_name(raw)
name = regexprep(raw, '\W', '_');
if isempty(name) || ~isletter(name(1))
    name = ['case_' name];
end
end
