"""
Distributed Data Parallel training via per-parameter gradient all-reduce.

Each parameter's gradient is individually all-reduced (asynchronously) as
soon as it is ready during the backward pass. The caller must invoke
``finish_gradient_synchronization`` (or equivalently
``ddp_individual_parameters_on_after_backward``) after the backward pass
and before the optimizer step to ensure all communication is complete and
gradients have been averaged.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List


class DDPIndividualParameters(nn.Module):
    """
    A minimal DDP wrapper that:
    1. Broadcasts all parameters from rank 0 to every other rank on construction,
       so all processes start with identical weights.
    2. Registers a backward hook on every parameter that requires grad. The hook
       launches an asynchronous ``all_reduce`` (sum) for that parameter's gradient
       as soon as the gradient is ready during the backward pass.
    3. Exposes ``finish_gradient_synchronization`` which waits for all pending
       communication ops and then divides each gradient by ``world_size`` to
       convert the sum into an average, matching single-process semantics.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self._world_size = dist.get_world_size()

        # Broadcast rank-0 parameters to all ranks so every process starts
        # from the same initial weights.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Pending async all_reduce handles, one per parameter that has grad.
        self._pending_work: List[dist.Work] = []

        # Register a backward hook for each parameter that requires grad.
        # We use register_post_accumulate_grad_hook (available in PyTorch >=2.1)
        # which fires after the gradient for that leaf tensor has been fully
        # accumulated. Fall back to register_hook on the tensor itself for
        # older PyTorch versions.
        self._hooks = []
        for param in self.module.parameters():
            if param.requires_grad:
                self._register_grad_hook(param)

    # ------------------------------------------------------------------
    # Hook registration helpers
    # ------------------------------------------------------------------

    def _register_grad_hook(self, param: torch.nn.Parameter) -> None:
        """Register an async all_reduce hook for *param*."""

        # ``register_post_accumulate_grad_hook`` is preferred because it fires
        # exactly once after gradient accumulation is finished for the parameter
        # (matching the DDP contract). It was added in PyTorch 2.1.
        if hasattr(param, "register_post_accumulate_grad_hook"):
            handle = param.register_post_accumulate_grad_hook(self._make_hook())
        else:
            # Older PyTorch: attach hook to the gradient tensor via autograd hook.
            handle = param.register_hook(self._make_tensor_hook())
        self._hooks.append(handle)

    def _make_hook(self):
        """Return a hook that launches an async all_reduce on param.grad."""

        def hook(param: torch.nn.Parameter) -> None:
            if param.grad is not None:
                work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                self._pending_work.append(work)

        return hook

    def _make_tensor_hook(self):
        """Return a hook for use with ``Tensor.register_hook`` (grad tensor)."""

        def hook(grad: torch.Tensor) -> None:
            work = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_work.append(work)

        return hook

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------

    def finish_gradient_synchronization(self) -> None:
        """
        Wait for all pending gradient all-reduces to complete, then
        divide every gradient by ``world_size`` to produce an average.

        Must be called after ``loss.backward()`` and before
        ``optimizer.step()``.
        """
        for work in self._pending_work:
            work.wait()
        self._pending_work.clear()

        # Average the gradients across ranks.
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.div_(self._world_size)

    # ------------------------------------------------------------------
    # nn.Module overrides
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
