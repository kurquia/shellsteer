"""Generate events with the Euler sampler of ShellFlow.

Flow time runs from t = 0 (noise) to t = 1 (data). `prior` draws the state at t = 0 from a seed. `generate`
integrates it to t = 1 in `steps` Euler steps and decodes the result to four-momenta.

ShellFlow's own sampler (`_generate_batch_rfm` in its training callback) runs without gradients. The gradient pass
needs them, so this module repeats its prior, Euler step, and decode, with the same operations in the same order.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from shellflow.models.riemannian_DiTs_geometry import MASSLESS_TYPES, exp_map_sphere_stable, minkowski_to_collator

from shellsteer.networks import Network

# A batch holds the event skeleton: types, charges, extra features, mask, and missing momentum.
# The network generates the kinematics; these keys only condition it.
BATCH_KEYS = ("particle_type", "particle_charge", "particle_extra", "particle_mask", "met_vector")


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
             grad: bool = False, on_step: Callable[[int], None] | None = None) -> torch.Tensor:
    """Integrate `state` from t = 0 to t = 1 and return the four-momenta (px, py, pz, E) in GeV, shape (B, N, 4).

    grad: keep the autograd graph through every step and the decode, for the gradient pass. Each step is
        recomputed during the backward pass instead of being stored.
    on_step: called with the step index k before the forward pass of step k. The steering edit uses it to know
        which xi(t_k) to add. It runs inside the recomputed region, so the backward pass sees the right k.
    """
    module, model = network.module, network.model
    if module.chart != "onshell" or module.direction_geometry != "S2":
        raise ValueError(f"only the onshell chart on S2 is supported, got {module.chart} on {module.direction_geometry}")
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

        state = torch.where(state.isnan(), torch.zeros_like(state), state)   # as ShellFlow's sampler
        return decode(model, state, batch["particle_type"], batch["particle_mask"], eps=1e-12 if grad else 0.0)


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
