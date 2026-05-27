from __future__ import annotations

from typing import Any

import torch

from . import AdjointOptions, NRConfig, SolveOptions


class CuPFFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        load_p: torch.Tensor,
        load_q: torch.Tensor,
        solver: Any,
        sbus_base_re: torch.Tensor,
        sbus_base_im: torch.Tensor,
        v0_va: torch.Tensor,
        v0_vm: torch.Tensor,
        config: Any | None = None,
        solve_options: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if config is None:
            config = NRConfig()
        if solve_options is None:
            solve_options = SolveOptions()
            solve_options.prepare_adjoint_cache = True
            solve_options.allow_explicit_transpose_fallback = True

        va = torch.empty_like(load_p)
        vm = torch.empty_like(load_q)
        solver.solve_with_adjoint_cache_torch(
            sbus_base_re,
            sbus_base_im,
            load_p.contiguous(),
            load_q.contiguous(),
            v0_va,
            v0_vm,
            va,
            vm,
            config,
            solve_options,
        )
        ctx.solver = solver
        return va, vm

    @staticmethod
    def backward(ctx: Any, grad_va: torch.Tensor, grad_vm: torch.Tensor) -> tuple[Any, ...]:
        grad_load_p = torch.empty_like(grad_va)
        grad_load_q = torch.empty_like(grad_vm)
        opts = AdjointOptions()
        opts.require_cached_factorization = True
        opts.allow_refactorize = False
        opts.allow_refactorize_for_backward = False
        opts.allow_explicit_transpose_fallback = True
        opts.check_residual = False
        ctx.solver.solve_adjoint_torch(
            grad_va.contiguous(),
            grad_vm.contiguous(),
            grad_load_p,
            grad_load_q,
            opts,
        )
        return grad_load_p, grad_load_q, None, None, None, None, None, None, None


def solve(load_p: torch.Tensor, load_q: torch.Tensor, solver: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return CuPFFunction.apply(
        load_p,
        load_q,
        solver,
        kwargs["sbus_base_re"],
        kwargs["sbus_base_im"],
        kwargs["v0_va"],
        kwargs["v0_vm"],
        kwargs.get("config"),
        kwargs.get("solve_options"),
    )
