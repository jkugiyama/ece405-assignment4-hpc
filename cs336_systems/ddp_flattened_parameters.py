"""
Distributed Data Parallel training via a single flattened gradient all-reduce.

This wrapper broadcasts parameters on construction, then after backward pass it
packs all parameter gradients into one flat tensor, issues a single all-reduce,
averages, and scatters the reduced gradients back to original tensors.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class DDPFlattenedParameters(nn.Module):
    """Minimal DDP wrapper that synchronizes gradients with one flat all-reduce."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self._world_size = dist.get_world_size()
        self._params = [p for p in self.module.parameters() if p.requires_grad]

        # Broadcast rank-0 parameters to all ranks for identical initialization.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def finish_gradient_synchronization(self) -> None:
        """
        Flatten gradients, all-reduce once, average, then unflatten back.

        Must be called after ``loss.backward()`` and before ``optimizer.step()``.
        """
        if not self._params:
            return

        has_grad: list[bool] = []
        flat_chunks: list[torch.Tensor] = []

        for p in self._params:
            if p.grad is None:
                has_grad.append(False)
                flat_chunks.append(torch.zeros_like(p).reshape(-1))
            else:
                has_grad.append(True)
                flat_chunks.append(p.grad.reshape(-1))

        flat_grads = torch.cat(flat_chunks, dim=0)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False)
        flat_grads.div_(self._world_size)

        offset = 0
        for p, p_has_grad in zip(self._params, has_grad):
            numel = p.numel()
            if p_has_grad:
                p.grad.copy_(flat_grads[offset : offset + numel].view_as(p))
            offset += numel

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
