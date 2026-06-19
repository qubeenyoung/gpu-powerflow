function manifest = export_exapf_island_numeric_cases(case_path, output_dir, matpower_home)
%EXPORT_EXAPF_ISLAND_NUMERIC_CASES Export one ExaPF-compatible file per island.
%
% ExaPF currently assumes exactly one REF bus in a PowerNetwork. MATPOWER cases
% such as case_SyntheticUSA contain multiple disconnected islands, each with its
% own REF bus. This exporter preserves that model by splitting the MATPOWER case
% with extract_islands(), then writing each island as a numeric-only MATPOWER v2
% file after ext2int().

if nargin < 3 || isempty(matpower_home)
    matpower_home = getenv_with_default('MATPOWER_HOME', '/opt/matpower');
end
if nargin < 2 || isempty(output_dir)
    error('export_exapf_island_numeric_cases:MissingOutputDir', 'output_dir is required');
end
if nargin < 1 || isempty(case_path)
    error('export_exapf_island_numeric_cases:MissingCasePath', 'case_path is required');
end

setup_matpower_path(matpower_home);
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

[~, case_name, ~] = fileparts(case_path);
mpc = loadcase(case_path);
islands = extract_islands(mpc);
if ~iscell(islands)
    islands = num2cell(islands);
end

manifest_path = fullfile(output_dir, 'cases.txt');
manifest_fid = fopen(manifest_path, 'w');
if manifest_fid < 0
    error('export_exapf_island_numeric_cases:OpenFailed', 'Could not open %s', manifest_path);
end
cleanup = onCleanup(@() fclose(manifest_fid)); %#ok<NASGU>

manifest = struct('case_count', numel(islands), 'output_dir', output_dir, 'cases_file', manifest_path);
for k = 1:numel(islands)
    island = ext2int(islands{k});
    island = ensure_single_ref(island);
    island_name = sprintf('%s_island%d', case_name, k);
    out_path = fullfile(output_dir, [sanitize_function_name(island_name) '_exapf_numeric.m']);
    write_numeric_case(out_path, island_name, island);
    fprintf(manifest_fid, '%s\n', out_path);
    fprintf('[exapf-island-export][OK] %s buses=%d -> %s\n', island_name, size(island.bus, 1), out_path);
end
end

function setup_matpower_path(matpower_home)
if exist(matpower_home, 'dir') ~= 7
    error('export_exapf_island_numeric_cases:MatpowerHomeNotFound', 'MATPOWER_HOME not found: %s', matpower_home);
end
addpath(genpath(matpower_home));
if exist('loadcase', 'file') ~= 2 || exist('extract_islands', 'file') ~= 2
    error('export_exapf_island_numeric_cases:MatpowerNotFound', 'MATPOWER loadcase/extract_islands was not found');
end
end

function mpc = ensure_single_ref(mpc)
REF = 3;
PV = 2;
BUS_TYPE = 2;
GEN_BUS = 1;
GEN_STATUS = 8;

ref_rows = find(mpc.bus(:, BUS_TYPE) == REF);
if numel(ref_rows) == 1
    return;
end
if numel(ref_rows) > 1
    mpc.bus(ref_rows(2:end), BUS_TYPE) = PV;
    return;
end

active_gen_buses = mpc.gen(mpc.gen(:, GEN_STATUS) > 0, GEN_BUS);
pv_rows = find(mpc.bus(:, BUS_TYPE) == PV & ismember(mpc.bus(:, 1), active_gen_buses));
if isempty(pv_rows)
    error('export_exapf_island_numeric_cases:NoReferenceCandidate', 'Island has no REF bus and no active PV generator bus');
end
mpc.bus(pv_rows(1), BUS_TYPE) = REF;
end

function write_numeric_case(out_path, case_name, mpc)
fid = fopen(out_path, 'w');
if fid < 0
    error('export_exapf_island_numeric_cases:OpenFailed', 'Could not open %s', out_path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

function_name = sanitize_function_name([case_name '_exapf_numeric']);
fprintf(fid, 'function mpc = %s\n', function_name);
fprintf(fid, '%% Generated from original MATPOWER case via loadcase(), extract_islands(), ext2int().\n');
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
