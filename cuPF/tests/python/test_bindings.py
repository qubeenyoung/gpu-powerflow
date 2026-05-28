from __future__ import annotations

import importlib
import math

import numpy as np


def import_cupf():
    try:
        return importlib.import_module("cupf")
    except Exception:
        return importlib.import_module("_cupf")


def two_bus_case():
    y = 1.0 - 5.0j
    slack_v = 1.0 + 0.0j
    pq_v = 0.97 - 0.05j
    pq_current = y * (pq_v - slack_v)
    pq_sbus = pq_v * np.conjugate(pq_current)
    return {
        "indptr": np.array([0, 2, 4], dtype=np.int32),
        "indices": np.array([0, 1, 0, 1], dtype=np.int32),
        "data": np.array([y, -y, -y, y], dtype=np.complex128),
        "sbus": np.array([0.0 + 0.0j, pq_sbus], dtype=np.complex128),
        "v0": np.array([slack_v, 1.0 + 0.0j], dtype=np.complex128),
        "expected_v": np.array([slack_v, pq_v], dtype=np.complex128),
        "pv": np.array([], dtype=np.int32),
        "pq": np.array([1], dtype=np.int32),
    }


def main() -> None:
    cupf = import_cupf()

    options = cupf.NewtonOptions()
    assert options.backend == cupf.BackendKind.CPU
    assert options.compute == cupf.ComputePolicy.FP64
    options.compute = cupf.ComputePolicy.FP32
    assert options.compute == cupf.ComputePolicy.FP32
    options.compute = cupf.ComputePolicy.FP64

    data = two_bus_case()
    solver = cupf.NewtonSolver(options)
    assert not hasattr(solver, "solve_adjoint_cuda_raw_unsafe")
    solver.initialize(
        data["indptr"],
        data["indices"],
        data["data"],
        2,
        2,
        data["pv"],
        data["pq"],
    )

    config = cupf.NRConfig()
    config.tolerance = 1e-10
    config.max_iter = 20
    result = solver.solve(
        data["indptr"],
        data["indices"],
        data["data"],
        2,
        2,
        data["sbus"],
        data["v0"],
        data["pv"],
        data["pq"],
        config,
    )

    assert result.converged
    assert math.isfinite(result.final_mismatch)
    np.testing.assert_allclose(result.V_numpy, data["expected_v"], rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    main()
