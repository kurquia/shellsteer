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


def check_width(h: torch.Tensor, width: int, what: str) -> None:
    if h.shape[-1] != width:
        raise ValueError(f"{what} has width {width}, the residual stream of this block has width {h.shape[-1]}")


class Probe:
    """Adds `delta[k]` to the two tokens of every pair at step k; `delta` is zero and records the gradient.

    minus, plus: (B,) slot of the negative and the positive particle of each pair.
    After backward, delta.grad has shape (steps, B, 2, width), negative token first; width is the network's.
    """

    def __init__(self, network: Network, steps: int, minus: torch.Tensor, plus: torch.Tensor) -> None:
        self.rows = torch.arange(len(minus), device=minus.device)
        self.minus, self.plus = minus, plus
        self.width = network.model.d_model
        self.delta = torch.zeros(steps, len(minus), 2, self.width, device=minus.device, requires_grad=True)
        self.step = 0

    def set_step(self, k: int) -> None:
        self.step = k

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        check_width(h, self.width, "the probe")
        d = self.delta[self.step].to(h.dtype)
        out = h.clone()
        out[self.rows, self.minus] = h[self.rows, self.minus] + d[:, 0]
        out[self.rows, self.plus] = h[self.rows, self.plus] + d[:, 1]
        return out


class Steer:
    """Adds alpha xi^-(t_k) to the tokens in `edit_minus` and alpha xi^+(t_k) to the tokens in `edit_plus`.

    xi: (steps, 2, width), negative half first.
    edit_minus, edit_plus: (B, N), True on the tokens to edit, from `events.edited_tokens`. The config option
        `tokens` decides them: the pair's two tokens, one of them, or every token of the type with that charge.
    steps: edit only at Euler steps first..last (inclusive); None edits every step.
    norm_restore: rescale each edited token to the norm it had before the edit.
    """

    def __init__(self, xi: torch.Tensor, alpha: float, edit_minus: torch.Tensor, edit_plus: torch.Tensor, *,
                 steps: tuple[int, int] | None = None, norm_restore: bool = False) -> None:
        if xi.dim() != 3 or xi.shape[1] != 2:
            raise ValueError(f"xi has shape {tuple(xi.shape)}, expected (steps, 2, width)")
        if (edit_minus & edit_plus).any():
            raise ValueError("a token is in both edit_minus and edit_plus")
        self.xi, self.alpha = xi, float(alpha)
        self.edit_minus, self.edit_plus = edit_minus.unsqueeze(-1), edit_plus.unsqueeze(-1)
        self.steps, self.norm_restore = steps, norm_restore
        self.step = 0

    def set_step(self, k: int) -> None:
        if k >= self.xi.shape[0]:
            raise ValueError(f"step {k}, but xi has only {self.xi.shape[0]} steps")
        self.step = k

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        check_width(h, self.xi.shape[-1], "xi")
        if self.steps is not None and not self.steps[0] <= self.step <= self.steps[1]:
            return h
        xi = self.xi[self.step].to(h.dtype)
        out = torch.where(self.edit_minus, h + self.alpha * xi[0], h)
        out = torch.where(self.edit_plus, out + self.alpha * xi[1], out)
        if self.norm_restore:
            edited = self.edit_minus | self.edit_plus
            scale = h.norm(dim=-1, keepdim=True) / out.norm(dim=-1, keepdim=True)
            out = torch.where(edited, out * scale, out)
        return out
