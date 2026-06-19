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
      --batch N              Number of identical systems to solve as a block. Default: 1.
      --max-block N           Cap the per-block size; a larger --batch is split into
                              ceil(batch/N) chunks (avoids GPU OOM on large grids).
                              0 = use batch as one block, auto-halving on OOM. Default: 0.
      --repeats N             Timed repetitions per case. Default: 5.
      --warmups N             Warmup repetitions per case. Default: 1.
      --output PATH           CSV output path.
      --continue-on-error     Skip cases that ExaPF cannot parse or solve.
      --help                  Show this help.
    """)
end

function parse_args(argv)
    opts = Dict{String,Any}(
        "cases" => String[],
        "data_root" => DEFAULT_DATA_ROOT,
        "backend" => "cuda",
        "batch" => 1,
        "max_block" => 0,
        "repeats" => 5,
        "warmups" => 1,
        "output" => "",
        "continue_on_error" => false,
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
        elseif arg == "--batch"
            i += 1
            i <= length(argv) || error("--batch requires a value")
            opts["batch"] = parse(Int, argv[i])
        elseif arg == "--max-block"
            i += 1
            i <= length(argv) || error("--max-block requires a value")
            opts["max_block"] = parse(Int, argv[i])
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
        elseif arg == "--continue-on-error"
            opts["continue_on_error"] = true
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
    if opts["batch"] < 1
        error("--batch must be at least 1")
    end
    if opts["max_block"] < 0
        error("--max-block must be non-negative (0 = no cap)")
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

function run_once(case_path::String, backend_name::String, backend, batch::Int)
    form = nothing
    stack = nothing
    setup_s = @elapsed begin
        batch == 1 || error("run_once only handles batch=1; use build_batch_instance for batched runs")
        polar = ExaPF.PolarForm(case_path, backend)
        form = polar
        stack = ExaPF.NetworkStack(form)
        synchronize_backend(backend_name)
    end

    conv = nothing
    run_s = @elapsed begin
        conv = ExaPF.run_pf(form, stack; verbose=0)
        synchronize_backend(backend_name)
    end

    return (
        nbus = form.network.nbus,
        batch = batch,
        converged = conv.has_converged,
        iterations = conv.n_iterations,
        residual = conv.norm_residuals,
        setup_ms = setup_s * 1000.0,
        wall_ms = run_s * 1000.0,
        total_ms = conv.time_total * 1000.0,
        total_ms_per_system = conv.time_total * 1000.0 / batch,
        jacobian_ms = conv.time_jacobian * 1000.0,
        solver_update_ms = conv.time_linear_solver_update * 1000.0,
        solver_ldiv_ms = conv.time_linear_solver_ldiv * 1000.0,
    )
end

function build_batch_instance(case_path::String, backend_name::String, backend, batch::Int)
    polar = nothing
    block_form = nothing
    stack = nothing
    jacobian = nothing
    linear_solver = nothing

    setup_s = @elapsed begin
        polar = ExaPF.load_polar(case_path, backend)

        pload = Array(ExaPF.PS.get(polar.network, ExaPF.PS.ActiveLoad()))
        qload = Array(ExaPF.PS.get(polar.network, ExaPF.PS.ReactiveLoad()))
        ploads = repeat(pload, 1, batch)
        qloads = repeat(qload, 1, batch)

        block_form = ExaPF.BlockPolarForm(polar, batch)
        stack = ExaPF.NetworkStack(block_form)
        ExaPF.set_params!(stack, ploads, qloads)

        powerflow = ExaPF.PowerFlowBalance(block_form) ∘ ExaPF.Basis(block_form)
        jacobian = ExaPF.BatchJacobian(block_form, powerflow, ExaPF.State())
        ExaPF.set_params!(jacobian, stack)
        ExaPF.jacobian!(jacobian, stack)
        synchronize_backend(backend_name)

        linear_solver = ExaPF.LinearSolvers.default_linear_solver(jacobian.J; nblocks=batch)
        synchronize_backend(backend_name)
    end

    return (
        case_path = case_path,
        nbus = ExaPF.PS.get(polar.network, ExaPF.PS.NumberOfBuses()),
        batch = batch,
        backend_name = backend_name,
        setup_ms = setup_s * 1000.0,
        form = block_form,
        stack = stack,
        jacobian = jacobian,
        linear_solver = linear_solver,
    )
end

function run_batch_once(instance)
    conv = nothing
    run_s = @elapsed begin
        ExaPF.init!(instance.form, instance.stack)
        synchronize_backend(instance.backend_name)
        solver = NewtonRaphson(; verbose=0, tol=1e-8, maxiter=20)
        conv = ExaPF.nlsolve!(solver, instance.jacobian, instance.stack; linear_solver=instance.linear_solver)
        synchronize_backend(instance.backend_name)
    end

    return (
        nbus = instance.nbus,
        batch = instance.batch,
        converged = conv.has_converged,
        iterations = conv.n_iterations,
        residual = conv.norm_residuals,
        setup_ms = instance.setup_ms,
        wall_ms = run_s * 1000.0,
        total_ms = conv.time_total * 1000.0,
        total_ms_per_system = conv.time_total * 1000.0 / instance.batch,
        jacobian_ms = conv.time_jacobian * 1000.0,
        solver_update_ms = conv.time_linear_solver_update * 1000.0,
        solver_ldiv_ms = conv.time_linear_solver_ldiv * 1000.0,
    )
end

# Memory-safe batching.
#
# The batched power flow does NOT run out of room in the (efficient) cuDSS batched
# linear solve — it runs out in ExaPF's batched ForwardDiff Jacobian, which
# allocates `Dual{Float64, ncolors}` arrays sized O(block × network × ncolors).
# On a large grid this exceeds GPU memory once `block` grows (e.g. case9241pegase
# OOMs around block 256 on a 24 GB card). Since the batch systems are identical,
# we solve `batch` of them as ceil(batch / block) chunks of one `block`-sized
# instance: pick the largest block that fits (halving on out-of-memory), then
# replay it. Peak memory is bounded by `block`, not by the full `batch`.
function prepare_batched(case_path, backend_name, backend, batch::Int, max_block::Int)
    block = max_block > 0 ? min(batch, max_block) : batch
    while true
        try
            inst = build_batch_instance(case_path, backend_name, backend, block)
            return (instance = inst, block = block, nchunks = cld(batch, block), batch = batch)
        catch err
            is_oom = err isa CUDA.OutOfGPUMemoryError ||
                     occursin("Out of GPU memory", sprint(showerror, err))
            if is_oom && block > 1
                backend_name == "cuda" && CUDA.reclaim()
                block = max(1, block ÷ 2)
                @warn "ExaPF batch: block exceeded GPU memory; retrying with block=$block"
            else
                rethrow()
            end
        end
    end
end

# Replay the block-sized instance `nchunks` times to cover the full batch,
# aggregating the per-phase timings. Per-system metrics use the logical `batch`.
function run_batched_once(prep)
    inst = prep.instance
    converged, iters, residual = true, 0, 0.0
    wall, total, jac, upd, ldiv = 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in 1:prep.nchunks
        r = run_batch_once(inst)
        wall += r.wall_ms; total += r.total_ms
        jac += r.jacobian_ms; upd += r.solver_update_ms; ldiv += r.solver_ldiv_ms
        converged &= r.converged; iters = max(iters, r.iterations); residual = max(residual, r.residual)
    end
    return (
        nbus = inst.nbus,
        batch = prep.batch,
        converged = converged,
        iterations = iters,
        residual = residual,
        setup_ms = inst.setup_ms,
        wall_ms = wall,
        total_ms = total,
        total_ms_per_system = total / prep.batch,
        jacobian_ms = jac,
        solver_update_ms = upd,
        solver_ldiv_ms = ldiv,
    )
end

function write_header(io)
    println(io, join([
        "case",
        "path",
        "backend",
        "run",
        "warmup",
        "batch",
        "buses",
        "converged",
        "iterations",
        "residual",
        "setup_ms",
        "wall_ms",
        "time_total_ms",
        "time_total_ms_per_system",
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
        string(result.batch),
        string(result.nbus),
        string(result.converged),
        string(result.iterations),
        @sprintf("%.12e", result.residual),
        @sprintf("%.6f", result.setup_ms),
        @sprintf("%.6f", result.wall_ms),
        @sprintf("%.6f", result.total_ms),
        @sprintf("%.6f", result.total_ms_per_system),
        @sprintf("%.6f", result.jacobian_ms),
        @sprintf("%.6f", result.solver_update_ms),
        @sprintf("%.6f", result.solver_ldiv_ms),
    ]
    println(io, join(values, ","))
end

function summarize(case_results)
    println()
    println("Summary over timed runs")
    println("case,buses,batch,converged,repeats,total_ms_mean,total_ms_per_system_mean,total_ms_min,wall_ms_mean,wall_ms_min")
    for (name, rows) in case_results
        totals = [r.total_ms for r in rows]
        per_system = [r.total_ms_per_system for r in rows]
        walls = [r.wall_ms for r in rows]
        buses = first(rows).nbus
        batch = first(rows).batch
        converged = all(r.converged for r in rows)
        @printf(
            "%s,%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            name,
            buses,
            batch,
            string(converged),
            length(rows),
            mean(totals),
            mean(per_system),
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

    mkpath(dirname(opts["output"]))
    timed_results = Pair{String,Vector{Any}}[]

    open(opts["output"], "w") do io
        write_header(io)
        for case_arg in opts["cases"]
            try
                case_path = materialize_parser_friendly_case(resolve_case(case_arg, opts["data_root"]))
                name = case_name(case_path)
                println("== $name ($backend_name) ==")
                prep = opts["batch"] == 1 ? nothing :
                    prepare_batched(case_path, backend_name, backend, opts["batch"], opts["max_block"])
                if prep !== nothing && prep.nchunks > 1
                    @printf("batch %d split into %d chunk(s) of block %d (GPU-memory safe)\n",
                        prep.batch, prep.nchunks, prep.block)
                end

                for w in 1:opts["warmups"]
                    result = opts["batch"] == 1 ? run_once(case_path, backend_name, backend, opts["batch"]) : run_batched_once(prep)
                    write_row(io, case_path, backend_name, w, true, result)
                    @printf("warmup %d: converged=%s iter=%d total=%.3f ms wall=%.3f ms\n",
                        w, string(result.converged), result.iterations, result.total_ms, result.wall_ms)
                    flush(io)
                end

                rows = Any[]
                for r in 1:opts["repeats"]
                    result = opts["batch"] == 1 ? run_once(case_path, backend_name, backend, opts["batch"]) : run_batched_once(prep)
                    push!(rows, result)
                    write_row(io, case_path, backend_name, r, false, result)
                    @printf("run %d: converged=%s iter=%d total=%.3f ms wall=%.3f ms\n",
                        r, string(result.converged), result.iterations, result.total_ms, result.wall_ms)
                    flush(io)
                end
                push!(timed_results, name => rows)
            catch err
                if opts["continue_on_error"]
                    println(stderr, "[exapf][SKIP] case=$(case_arg) error=$(sprint(showerror, err))")
                    flush(stderr)
                    continue
                end
                rethrow()
            end
        end
    end

    summarize(timed_results)
    println()
    println("Wrote CSV: $(opts["output"])")
end

main()
