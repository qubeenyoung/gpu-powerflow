#!/usr/bin/env julia

using Dates
using Printf
using Statistics
using Pkg.Artifacts

using ExaPF
using KernelAbstractions
using CUDA
using CUDSS

const DEFAULT_CASES = ["case3120sp", "case6470rte", "case8387pegase"]
const DEFAULT_DATA_ROOT = "/datasets/matpower/raw"

function usage()
    println("""
    Usage:
      julia --project=exapf exapf/run_powerflow.jl [options]

    Options:
      --case NAME_OR_PATH      Add one MATPOWER case. May be repeated.
      --cases A,B,C           Set comma-separated MATPOWER cases.
      --data-root PATH        Directory used to resolve case names.
      --backend cuda|cpu      Execution backend. Default: cuda.
      --repeats N             Timed repetitions per case. Default: 5.
      --warmups N             Warmup repetitions per case. Default: 1.
      --output PATH           CSV output path.
      --help                  Show this help.
    """)
end

function parse_args(argv)
    opts = Dict{String,Any}(
        "cases" => String[],
        "data_root" => DEFAULT_DATA_ROOT,
        "backend" => "cuda",
        "repeats" => 5,
        "warmups" => 1,
        "output" => "",
    )

    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--case"
            i += 1
            i <= length(argv) || error("--case requires a value")
            push!(opts["cases"], argv[i])
        elseif arg == "--cases"
            i += 1
            i <= length(argv) || error("--cases requires a value")
            opts["cases"] = filter(!isempty, split(argv[i], ","))
        elseif arg == "--data-root"
            i += 1
            i <= length(argv) || error("--data-root requires a value")
            opts["data_root"] = argv[i]
        elseif arg == "--backend"
            i += 1
            i <= length(argv) || error("--backend requires a value")
            opts["backend"] = lowercase(argv[i])
        elseif arg == "--repeats"
            i += 1
            i <= length(argv) || error("--repeats requires a value")
            opts["repeats"] = parse(Int, argv[i])
        elseif arg == "--warmups"
            i += 1
            i <= length(argv) || error("--warmups requires a value")
            opts["warmups"] = parse(Int, argv[i])
        elseif arg == "--output"
            i += 1
            i <= length(argv) || error("--output requires a value")
            opts["output"] = argv[i]
        else
            error("unknown argument: $arg")
        end
        i += 1
    end

    if isempty(opts["cases"])
        opts["cases"] = copy(DEFAULT_CASES)
    end
    if opts["repeats"] < 1
        error("--repeats must be at least 1")
    end
    if opts["warmups"] < 0
        error("--warmups must be non-negative")
    end
    if !(opts["backend"] in ("cuda", "cpu"))
        error("--backend must be 'cuda' or 'cpu'")
    end
    if isempty(opts["output"])
        stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        opts["output"] = joinpath(@__DIR__, "results", "exapf_powerflow_$stamp.csv")
    end

    return opts
end

function resolve_case(case_arg::AbstractString, data_root::AbstractString)
    if isfile(case_arg)
        return abspath(case_arg)
    end

    filename = endswith(case_arg, ".m") ? case_arg : "$case_arg.m"
    rooted = joinpath(data_root, filename)
    if isfile(rooted)
        return abspath(rooted)
    end

    artifacts_toml = joinpath(pkgdir(ExaPF), "Artifacts.toml")
    exadata_root = joinpath(ensure_artifact_installed("ExaData", artifacts_toml), "ExaData")
    artifact_path = joinpath(exadata_root, filename)
    if isfile(artifact_path)
        return abspath(artifact_path)
    end

    error("could not resolve case '$case_arg' in '$data_root' or ExaPF ExaData")
end

function materialize_parser_friendly_case(case_path::String)
    if basename(case_path) != "case8387pegase.m"
        return case_path
    end

    out_dir = joinpath(@__DIR__, "generated")
    out_path = joinpath(out_dir, "case8387pegase_exapf.m")
    source_mtime = mtime(case_path)
    if isfile(out_path) && mtime(out_path) >= source_mtime
        return out_path
    end

    mkpath(out_dir)
    open(out_path, "w") do io
        skipping_fixed_block = false
        for line in eachline(case_path)
            stripped = strip(line)
            if startswith(stripped, "fixed =")
                continue
            end
            if occursin(r"^if\s+fixed\b", stripped)
                skipping_fixed_block = true
                continue
            end
            if skipping_fixed_block
                if stripped == "end"
                    skipping_fixed_block = false
                end
                continue
            end
            println(io, line)
        end
    end
    println("Prepared ExaPF parser-friendly case: $out_path")
    return out_path
end

function make_backend(name::AbstractString)
    if name == "cuda"
        CUDA.functional() || error("CUDA.jl is not functional on this machine")
        return CUDABackend()
    end
    return CPU()
end

function synchronize_backend(name::AbstractString)
    name == "cuda" && CUDA.synchronize()
    return nothing
end

case_name(path::AbstractString) = splitext(basename(path))[1]

function run_once(case_path::String, backend_name::String, backend)
    polar = nothing
    stack = nothing
    setup_s = @elapsed begin
        polar = ExaPF.PolarForm(case_path, backend)
        stack = ExaPF.NetworkStack(polar)
        synchronize_backend(backend_name)
    end

    conv = nothing
    run_s = @elapsed begin
        conv = ExaPF.run_pf(polar, stack; verbose=0)
        synchronize_backend(backend_name)
    end

    return (
        nbus = polar.network.nbus,
        converged = conv.has_converged,
        iterations = conv.n_iterations,
        residual = conv.norm_residuals,
        setup_ms = setup_s * 1000.0,
        wall_ms = run_s * 1000.0,
        total_ms = conv.time_total * 1000.0,
        jacobian_ms = conv.time_jacobian * 1000.0,
        solver_update_ms = conv.time_linear_solver_update * 1000.0,
        solver_ldiv_ms = conv.time_linear_solver_ldiv * 1000.0,
    )
end

function write_header(io)
    println(io, join([
        "case",
        "path",
        "backend",
        "run",
        "warmup",
        "buses",
        "converged",
        "iterations",
        "residual",
        "setup_ms",
        "wall_ms",
        "time_total_ms",
        "time_jacobian_ms",
        "time_linear_solver_update_ms",
        "time_linear_solver_ldiv_ms",
    ], ","))
end

function write_row(io, case_path, backend_name, run_index, is_warmup, result)
    values = [
        case_name(case_path),
        case_path,
        backend_name,
        string(run_index),
        string(is_warmup),
        string(result.nbus),
        string(result.converged),
        string(result.iterations),
        @sprintf("%.12e", result.residual),
        @sprintf("%.6f", result.setup_ms),
        @sprintf("%.6f", result.wall_ms),
        @sprintf("%.6f", result.total_ms),
        @sprintf("%.6f", result.jacobian_ms),
        @sprintf("%.6f", result.solver_update_ms),
        @sprintf("%.6f", result.solver_ldiv_ms),
    ]
    println(io, join(values, ","))
end

function summarize(case_results)
    println()
    println("Summary over timed runs")
    println("case,buses,converged,repeats,total_ms_mean,total_ms_min,wall_ms_mean,wall_ms_min")
    for (name, rows) in case_results
        totals = [r.total_ms for r in rows]
        walls = [r.wall_ms for r in rows]
        buses = first(rows).nbus
        converged = all(r.converged for r in rows)
        @printf(
            "%s,%d,%s,%d,%.6f,%.6f,%.6f,%.6f\n",
            name,
            buses,
            string(converged),
            length(rows),
            mean(totals),
            minimum(totals),
            mean(walls),
            minimum(walls),
        )
    end
end

function main()
    opts = parse_args(ARGS)
    backend_name = opts["backend"]
    backend = make_backend(backend_name)
    case_paths = [
        materialize_parser_friendly_case(resolve_case(c, opts["data_root"]))
        for c in opts["cases"]
    ]

    mkpath(dirname(opts["output"]))
    timed_results = Pair{String,Vector{Any}}[]

    open(opts["output"], "w") do io
        write_header(io)
        for case_path in case_paths
            name = case_name(case_path)
            println("== $name ($backend_name) ==")

            for w in 1:opts["warmups"]
                result = run_once(case_path, backend_name, backend)
                write_row(io, case_path, backend_name, w, true, result)
                @printf("warmup %d: converged=%s iter=%d total=%.3f ms wall=%.3f ms\n",
                    w, string(result.converged), result.iterations, result.total_ms, result.wall_ms)
                flush(io)
            end

            rows = Any[]
            for r in 1:opts["repeats"]
                result = run_once(case_path, backend_name, backend)
                push!(rows, result)
                write_row(io, case_path, backend_name, r, false, result)
                @printf("run %d: converged=%s iter=%d total=%.3f ms wall=%.3f ms\n",
                    r, string(result.converged), result.iterations, result.total_ms, result.wall_ms)
                flush(io)
            end
            push!(timed_results, name => rows)
        end
    end

    summarize(timed_results)
    println()
    println("Wrote CSV: $(opts["output"])")
end

main()
