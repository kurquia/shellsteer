"""Read and edit the residual stream at the output of a block, with PyTorch forward hooks.

A hook sees h_L, the output of model.blocks[L], of shape (B, N, width), and returns the edited stream. `attach`
registers it for the length of a `with` block. The hook learns the Euler step from `set_step`, which the sampler
calls through `generate(..., on_step=...)`.

Probe: adds zeros to the two tokens of the pair, one zero tensor per step. The events do not change, and after the
    backward pass the gradient of `probe.delta` at step k is G_p(t_k) of every pair.
Steer: adds alpha xi^-(t_k) to the negative token and alpha xi^+(t_k) to the positive token.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch

from shellsteer.networks import Network


@contextmanager
def attach(network: Network, block: int, hook: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    """Run `hook` on the output of `block` inside the `with` block; remove it on exit, also after an error."""

    def forward_hook(module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.dim() != 3:
            raise TypeError(f"block {block} returned {type(output).__name__}, expected a (B, N, width) tensor")
        return hook(output)

    handle = network.model.blocks[block].register_forward_hook(forward_hook)
    try:
        yield
    finally:
        handle.remove()


def on_step(*hooks) -> Callable[[int], None]:
    """One `on_step` callback for `generate` that sets the step of every hook."""
    def set_steps(k: int) -> None:
        for hook in hooks:
            hook.set_step(k)
    return set_steps


class Probe:
    """Adds `delta[k]` to the two tokens of every pair at step k; `delta` is zero and records the gradient.

    minus, plus: (B,) slot of the negative and the positive particle of each pair.
    After backward, delta.grad has shape (steps, B, 2, width), negative token first.
    """

    def __init__(self, steps: int, minus: torch.Tensor, plus: torch.Tensor, width: int) -> None:
        self.rows = torch.arange(len(minus), device=minus.device)
        self.minus, self.plus = minus, plus
        self.delta = torch.zeros(steps, len(minus), 2, width, device=minus.device, requires_grad=True)
        self.step = 0

    def set_step(self, k: int) -> None:
        self.step = k

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        d = self.delta[self.step].to(h.dtype)
        out = h.clone()
        out[self.rows, self.minus] = h[self.rows, self.minus] + d[:, 0]
        out[self.rows, self.plus] = h[self.rows, self.plus] + d[:, 1]
        return out


class Steer:
    """Adds alpha xi^-(t_k) to the tokens in `minus` and alpha xi^+(t_k) to the tokens in `plus`.

    xi: (steps, 2, width), negative half first.
    minus, plus: (B, N) boolean masks of the edited tokens. The config option `tokens` decides them: the pair's two
        tokens, one of them, or every token of the type with that charge.
    steps: edit only at Euler steps first..last (inclusive); None edits every step.
    norm_restore: rescale each edited token to the norm it had before the edit.
    """

    def __init__(self, xi: torch.Tensor, alpha: float, minus: torch.Tensor, plus: torch.Tensor, *,
                 steps: tuple[int, int] | None = None, norm_restore: bool = False) -> None:
        if xi.dim() != 3 or xi.shape[1] != 2:
            raise ValueError(f"xi has shape {tuple(xi.shape)}, expected (steps, 2, width)")
        if (minus & plus).any():
            raise ValueError("a token is in both the minus and the plus mask")
        self.xi, self.alpha = xi, float(alpha)
        self.minus, self.plus = minus.unsqueeze(-1), plus.unsqueeze(-1)
        self.steps, self.norm_restore = steps, norm_restore
        self.step = 0

    def set_step(self, k: int) -> None:
        if k >= self.xi.shape[0]:
            raise ValueError(f"step {k}, but xi has only {self.xi.shape[0]} steps")
        self.step = k

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        if self.steps is not None and not self.steps[0] <= self.step <= self.steps[1]:
            return h
        xi = self.xi[self.step].to(h.dtype)
        out = torch.where(self.minus, h + self.alpha * xi[0], h)
        out = torch.where(self.plus, out + self.alpha * xi[1], out)
        if self.norm_restore:
            edited = self.minus | self.plus
            scale = h.norm(dim=-1, keepdim=True) / out.norm(dim=-1, keepdim=True)
            out = torch.where(edited, out * scale, out)
        return out
