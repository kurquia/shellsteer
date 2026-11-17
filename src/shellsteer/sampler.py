"""Generate events with the Euler sampler of ShellFlow.

Flow time runs from t = 0 (noise) to t = 1 (data). `prior` draws the state at t = 0 from a seed. `generate`
integrates it to t = 1 in `steps` Euler steps and decodes the result to four-momenta. `decoded_mass` is the mass the
decode builds each jet, large-R jet, and hadronic tau from (paper Fig. 14).

ShellFlow's sampler sets every NaN coordinate of the final state to 0 before the decode. `generate` counts these NaNs
per event, and the NaN entries of the decoded four-momenta, and returns the counts with the four-momenta. A NaN that
appears at any step stays NaN to the last step, so the final state shows it. The decode clamps energies and masses,
so a finite final state gives finite four-momenta; an infinite direction coordinate gives NaN ones. `on_nan` chooses
what happens to a NaN: "set_to_zero" keeps ShellFlow's replacement, and the paper's events, bit for bit; "stop" raises
`NanError`.

ShellFlow's own sampler (`_generate_batch_rfm` in its training callback) runs without gradients. The gradient pass
needs them, so this module repeats its prior, Euler step, and decode, with the same operations in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from shellflow.models.riemannian_DiTs_geometry import MASSLESS_TYPES, exp_map_sphere_stable, minkowski_to_collator

from shellsteer.networks import Network

OnNan = Literal["set_to_zero", "stop"]


class NanError(RuntimeError):
    """A generation with on_nan = "stop" met a NaN."""


class Generated(NamedTuple):
    four_momenta: torch.Tensor      # (B, N, 4) px, py, pz, E in GeV
    nan_state: torch.Tensor         # (B,) int64 NaN coordinates of the final state per event, padding slots included,
                                    # counted before they are set to 0
    nan_four_momenta: torch.Tensor  # (B,) int64 NaN entries of the four-momenta per event, after the decode
    mass: torch.Tensor              # (B, N) float32 `decoded_mass` in GeV: NaN on massless types and padding


@dataclass(frozen=True)
class NanCounts:
    """The NaN counts of one pass over a sample, per event in file order. `metadata()` goes into the files it writes.

    counts: (E, 3) int64, per event the NaN coordinates of the final state, the NaN entries of the four-momenta, and
    1 if the mass of the pair is NaN, else 0. Events without a pair, as in paper Fig. 4, have (E, 2): no pair mass.
    """
    on_nan: OnNan
    event: np.ndarray               # (E,) index of the event in the combined dataset
    counts: np.ndarray

    @staticmethod
    def of_batch(generated: Generated, mass: torch.Tensor | None) -> np.ndarray:
        """(B, 3) counts of one batch, with `mass` the (B,) pair mass computed from `generated.four_momenta`, or (B, 2)
        for mass None, events without a pair."""
        columns = [generated.nan_state, generated.nan_four_momenta]
        if mass is not None:
            columns.append(mass.isnan().long())
        return torch.stack(columns, 1).cpu().numpy()

    def metadata(self) -> dict:
        state, four = self.counts[:, 0], self.counts[:, 1]
        metadata = {"sampler.on_nan": self.on_nan, "events": len(self.event),
                    "NaN coordinates of the final state": int(state.sum()),
                    "events with NaN in the final state": self.event[state > 0].tolist(),
                    "NaN entries of the four-momenta": int(four.sum()),
                    "events with NaN in the four-momenta": self.event[four > 0].tolist()}
        if self.counts.shape[1] == 3:
            mass = self.counts[:, 2]
            metadata |= {"NaN pair masses": int(mass.sum()), "events with NaN pair mass": self.event[mass > 0].tolist()}
        return metadata

    def text(self) -> str:
        """One line for the log."""
        state, four = self.counts[:, 0], self.counts[:, 1]
        line = (f"NaN: {int(state.sum()):,} final-state coordinates in {int((state > 0).sum()):,} events (set to 0), "
                f"{int(four.sum()):,} four-momentum entries in {int((four > 0).sum()):,} events")
        return line + (f", {int(self.counts[:, 2].sum()):,} pair masses" if self.counts.shape[1] == 3 else "")


def batch_seed(seed: int, index: int) -> int:
    """The noise seed of batch `index`: the paper's rule seed x 1,000,003 + batch index."""
    return seed * 1_000_003 + index


def prior(network: Network, batch: dict[str, torch.Tensor], seed: int) -> torch.Tensor:
    """State at t = 0, shape (B, N, 5), drawn from `seed` on the device of the batch.

    Coordinate 0 and coordinate 4 are Gaussian, coordinates 1-3 are uniform on the sphere. Coordinate 4 is 0 for
    massless types, and every coordinate is 0 on padding slots.
    """
    p_type, p_mask = batch["particle_type"], batch["particle_mask"]
    generator = torch.Generator(device=p_type.device).manual_seed(seed)
    sigma = network.module.sigma_prior
    b, n = p_type.shape

    s0 = torch.randn(b, n, device=p_type.device, generator=generator) * sigma
    v = torch.randn(b, n, 3, device=p_type.device, generator=generator)
    direction = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    s4 = torch.randn(b, n, device=p_type.device, generator=generator) * sigma

    massless = torch.zeros_like(p_mask, dtype=torch.bool)
    for code in MASSLESS_TYPES:
        massless |= p_type == code
    active = p_mask > 0.5
    s4 = torch.where(massless & active, torch.zeros_like(s4), s4)
    state = torch.cat([s0.unsqueeze(-1), direction, s4.unsqueeze(-1)], dim=-1)
    return state * active.unsqueeze(-1).float()


def generate(network: Network, batch: dict[str, torch.Tensor], state: torch.Tensor, steps: int, *,
             on_nan: OnNan = "set_to_zero", label: str = "the batch", grad: bool = False,
             on_step: Callable[[int], None] | None = None) -> Generated:
    """Integrate `state` from t = 0 to t = 1; return the four-momenta (px, py, pz, E) in GeV, shape (B, N, 4), the
    NaN counts of each event, and the decoder's mass of each particle of a massive type.

    on_nan: "set_to_zero" sets NaN coordinates of the final state to 0 before the decode, as ShellFlow's sampler;
        "stop" raises NanError, naming `label` and the counts, if the final state or the four-momenta hold a NaN.

    grad: keep the autograd graph through every step and the decode, for the gradient pass. Each step is
        recomputed during the backward pass instead of being stored.
    on_step: called with the step index k before the forward pass of step k. The steering edit uses it to know
        which xi(t_k) to add. It runs inside the recomputed region, so the backward pass sees the right k.
    """
    module, model = network.module, network.model
    if on_nan not in ("set_to_zero", "stop"):
        raise ValueError(f"on_nan: {on_nan!r} is not one of 'set_to_zero', 'stop'")
    if module.chart != "onshell" or module.direction_geometry != "S2":
        raise ValueError(f"only the onshell chart on S2 is supported, "
                         f"got {module.chart} on {module.direction_geometry}")
    if state.shape != (*batch["particle_type"].shape, 5):
        raise ValueError(f"state has shape {tuple(state.shape)}, expected {(*batch['particle_type'].shape, 5)}")

    with torch.set_grad_enabled(grad):
        active = (batch["particle_mask"] > 0.5).unsqueeze(-1).float()
        b = state.shape[0]
        dt = 1.0 / steps
        for k in range(steps):
            t_k = min(max(k * dt, module.t_min), module.t_max)   # the network was trained on [t_min, t_max]

            def velocity(s, k=k, t_k=t_k):
                if on_step is not None:
                    on_step(k)
                t = torch.full((b,), t_k, device=s.device)
                return model(s, batch["particle_type"], batch["particle_charge"], batch["particle_extra"], t,
                             batch["particle_mask"], met_vector=batch["met_vector"])

            v = torch.utils.checkpoint.checkpoint(velocity, state, use_reentrant=False) if grad else velocity(state)
            s0 = state[..., 0] + dt * v[..., 0]
            direction = exp_map_sphere_stable(state[..., 1:4], dt * v[..., 1:4])
            s4 = state[..., 4] + dt * v[..., 4]
            state = torch.cat([s0.unsqueeze(-1), direction, s4.unsqueeze(-1)], dim=-1) * active

        nan_state = state.isnan().flatten(1).sum(1)
        if on_nan == "stop" and bool(nan_state.any()):
            raise NanError(f"{label}: {int(nan_state.sum()):,} NaN coordinates in the final state of "
                           f"{int((nan_state > 0).sum()):,} of {b:,} events (sampler.on_nan is stop)")
        state = torch.where(state.isnan(), torch.zeros_like(state), state)   # as ShellFlow's sampler
        four_momenta = decode(model, state, batch["particle_type"], batch["particle_mask"], eps=1e-12 if grad else 0.0)
        nan_four_momenta = four_momenta.isnan().flatten(1).sum(1)
        if on_nan == "stop" and bool(nan_four_momenta.any()):
            raise NanError(f"{label}: {int(nan_four_momenta.sum()):,} NaN entries in the four-momenta of "
                           f"{int((nan_four_momenta > 0).sum()):,} of {b:,} events (sampler.on_nan is stop)")
        mass = decoded_mass(model, state.detach(), batch["particle_type"], batch["particle_mask"])
        return Generated(four_momenta, nan_state, nan_four_momenta, mass)


def decode(model: torch.nn.Module, state: torch.Tensor, p_type: torch.Tensor, p_mask: torch.Tensor,
           eps: float = 0.0) -> torch.Tensor:
    """Four-momenta (px, py, pz, E) from a state on the manifold, as ShellFlow's `_decode_rfm_output`.

    Massless: E = E_ref exp(s0), |p| = sqrt(E^2 - m_rest^2). Massive: chi = s0 sigma + mu, m = exp(s4 sigma + mu),
    E = m cosh(chi), |p| = m sinh(chi). eps > 0 keeps the gradient of the square root finite at E = m_rest.
    """
    type_idx = model.map_type_to_idx(p_type)
    is_massless = model._is_massless_lut[type_idx]
    is_massive = model._is_massive_lut[type_idx]
    active = p_mask > 0.5

    e_ref = model._e_ref_lut[type_idx]
    m_rest = model._m_rest_lut[type_idx]
    chi_mu = model._chi_mu_lut[type_idx]
    chi_sigma = model._chi_sigma_lut[type_idx].clamp(min=0.01)
    logm_mu = model._logm_mu_lut[type_idx]
    logm_sigma = model._logm_sigma_lut[type_idx].clamp(min=0.01)
    n_hat = F.normalize(state[..., 1:4], dim=-1, eps=1e-6)

    e_massless = e_ref * torch.exp(state[..., 0].clamp(-10, 10))
    p_massless = (e_massless * e_massless - m_rest * m_rest).clamp(min=eps).sqrt().unsqueeze(-1) * n_hat

    chi = (state[..., 0] * chi_sigma + chi_mu).clamp(0, 12)
    mass = torch.exp((state[..., 4] * logm_sigma + logm_mu).clamp(-10, 10))
    e_massive = mass * torch.cosh(chi)
    p_massive = (mass * torch.sinh(chi)).unsqueeze(-1) * n_hat

    energy = torch.where(is_massless, e_massless, e_massive)
    momentum = torch.where(is_massless.unsqueeze(-1), p_massless, p_massive)
    four = torch.cat([energy.unsqueeze(-1), momentum], dim=-1)
    real = ((is_massless | is_massive) & active).unsqueeze(-1)
    return minkowski_to_collator(torch.where(real, four, torch.zeros_like(four)))


def decoded_mass(model: torch.nn.Module, state: torch.Tensor, p_type: torch.Tensor, p_mask: torch.Tensor
                 ) -> torch.Tensor:
    """(B, N) mass m = exp(s4 sigma + mu) in GeV that `decode` builds a massive particle from, with the same clamps;
    NaN on massless types and padding.

    The mass column of paper Fig. 14 shows it for jets, large-R jets, and hadronic taus. sqrt(E^2 - |p|^2) of the
    float32 four-momenta loses the mass of a light, boosted particle, such as a hadronic tau, to cancellation.
    """
    type_idx = model.map_type_to_idx(p_type)
    logm_mu = model._logm_mu_lut[type_idx]
    logm_sigma = model._logm_sigma_lut[type_idx].clamp(min=0.01)
    mass = torch.exp((state[..., 4] * logm_sigma + logm_mu).clamp(-10, 10))
    massive = model._is_massive_lut[type_idx] & (p_mask > 0.5)
    return torch.where(massive, mass, torch.full_like(mass, float("nan")))
