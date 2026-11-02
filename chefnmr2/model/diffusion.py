# Started from code from https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/diffusion.py
# MIT License
# Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro

import torch
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange
from math import sqrt

from chefnmr2.model import score_model as score_models
from chefnmr2.model.geometry import center


class AtomDiffusion(Module):
    """Flow-matching sampler of atom coordinates with velocity prediction.

        Forward process:     x_t      = (1 - t) * x_0 + t * eps          (t = sigma in [0, 1]; 1 = noise, 0 = data)
        Network call:        v        = F_theta(x_t, t)
        Sampling SDE:        dX       = (v - gamma * score) dt + sqrt(2 * gamma * noise_scale) dW
                             (sampling runs backward in time: sigma decreases, dt < 0)
        score from v:        score    = ((1 - t) * v + x_t) / (-t)

    Sampling details:
      Time grid                 linear from 1 to sigma_min, then 0
      gamma                     sigma / (1 - sigma + 0.01)
      Coordinates               centered at every step and at the end
      Scaling                   the network works at target_sigma_data; outputs are scaled by
                                sigma_data / target_sigma_data (arxiv.org/abs/2410.10356)
      Classifier-free guidance  conditional and unconditional outputs mixed by guidance_scale
      Multiplicity              generate multiple samples per input condition in one batch
    """

    def __init__(
        self,
        score_model_args,
        embedder_args,
        sigma_min=0.001,
        noise_scale=1.0,
        guidance_scale=0.0,  # Guidance scale for classifier-free guidance (coordinates)
        sigma_data=None,
        target_sigma_data=1.8,
    ):
        super().__init__()
        score_model_name = score_model_args.model_name
        self.score_model = getattr(score_models, score_model_args.model_name)(
            **score_model_args[score_model_name],
            embedder_args=embedder_args,
        )

        # Hyperparameters
        self.sigma_min = sigma_min
        self.noise_scale = noise_scale
        self.guidance_scale = guidance_scale  # Guidance scale for classifier-free guidance
        self.sigma_data = sigma_data
        # Faster DiT trick: https://arxiv.org/abs/2410.10356
        self.target_sigma_data = (
            target_sigma_data if target_sigma_data is not None else self.sigma_data
        )  # Scale inputs

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def float_to_tensor(self, value, batch_size, device):
        """Convert a float value to a tensor of shape (batch_size,)."""
        if isinstance(value, float):
            value = torch.full((batch_size,), value, device=device)
        return value

    def pad_sigma(self, sigma, batch_size, device):
        """Pad sigma to match the shape of the model inputs."""
        sigma = self.float_to_tensor(sigma, batch_size, device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")
        return padded_sigma

    # noisy_atom_coords = a_sigma * atom_coords + b_sigma * gaussian_noise
    def a_sigma(self, sigma):
        return 1 - sigma

    def b_sigma(self, sigma):
        return sigma

    # neural_network_input = c_in * atom_coords
    def c_in(self, sigma):
        return torch.ones_like(sigma)

    def noised_coords_in_network(self, atom_coords, sigma):
        padded_sigma = self.pad_sigma(sigma, atom_coords.shape[0], atom_coords.device)
        return self.c_in(padded_sigma) * atom_coords

    def sigma_in_network(self, sigma):
        return sigma

    # the velocity target is c_sigma * atom_coords + d_sigma * noise = noise - atom_coords
    def c_sigma(self, sigma):
        return -torch.ones_like(sigma)

    def sample_sigma_schedule(self, num_sampling_steps):
        sigmas = torch.linspace(1.0, self.sigma_min, num_sampling_steps, device=self.device)
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    def sample_gamma_schedule(self, sigmas):
        gammas = sigmas / (1 - sigmas + 0.01)
        return gammas

    def sample(self, model_inputs, num_sampling_steps, multiplicity=1):
        """Sample `multiplicity` structures per input -> coordinates (B * multiplicity, N, 3).

        A thin wrapper around `_sample_impl` that scopes the static-shape cache: everything on the
        conditioning path is CONSTANT for the whole denoising loop, so `begin_static_shapes()` lets
        the peak embedder compute its output once and reuse it; the `finally` is what makes that
        safe, because a cache left armed after an exception would silently apply stale values to
        the next, possibly differently shaped batch.
        """
        sm = self.score_model
        # `armed`: begin() raises when the module is ALREADY armed by another in-flight loop;
        # release only what this call armed.
        armed = False
        try:
            sm.begin_static_shapes()
            armed = True
            return self._sample_impl(
                model_inputs=model_inputs,
                num_sampling_steps=num_sampling_steps,
                multiplicity=multiplicity,
            )
        finally:
            if armed:
                sm.end_static_shapes()

    def _sample_impl(self, model_inputs, num_sampling_steps, multiplicity=1):
        atom_mask = model_inputs["atom_mask"].repeat_interleave(multiplicity, 0)
        shape = (*atom_mask.shape, 3)

        sigmas = self.sample_sigma_schedule(num_sampling_steps)  # coord noise level, 1 = noise, 0 = data
        gammas = self.sample_gamma_schedule(sigmas)
        schedule_steps = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)

        # gradually denoise
        for sigma_curr, sigma_next, gamma in schedule_steps:
            atom_coords = center(atom_coords, atom_mask)

            sigma_curr, sigma_next, gamma = sigma_curr.item(), sigma_next.item(), gamma.item()

            eps = torch.randn(shape, device=self.device)

            atom_coords = self._sample_one_step(
                model_inputs=model_inputs,
                multiplicity=multiplicity,
                atom_coords=atom_coords,
                sigma_curr=sigma_curr,
                sigma_next=sigma_next,
                gamma=gamma,
                eps=eps,
            )

        # Final centering to post-process the generated structure
        atom_coords = center(atom_coords, atom_mask)

        # Faster DiT, scale outputs to match training noise scale
        atom_coords = atom_coords * (self.sigma_data / self.target_sigma_data)
        return atom_coords

    def _sample_one_step(self, model_inputs, multiplicity, atom_coords, sigma_curr, sigma_next, gamma, eps):
        """Perform one step of sampling. Returns the next atom coordinates.

        dX_t = v_t dt - gamma_t score_t dt + sqrt(2 * gamma_t * noise_scale) dW_t
        """
        dt = sigma_next - sigma_curr  # dt < 0

        with torch.no_grad():
            net_out = self.neural_network_forward(
                atom_coords,
                sigma_curr,
                network_condition_kwargs=dict(
                    multiplicity=multiplicity,
                    model_inputs=model_inputs,
                    guidance_scale=self.guidance_scale,
                ),
            )

        velocity = net_out["r_update"]

        score = self.predict_score(
            noisy_atom_coords=atom_coords,
            sigma=sigma_curr,
            velocity=velocity,
        )

        drift = (velocity - gamma * score) * dt
        diffusion = sqrt(2 * -dt * gamma * self.noise_scale) * eps

        atom_coords_next = atom_coords + drift + diffusion

        return atom_coords_next

    def neural_network_forward(self, noisy_atom_coords, sigma, network_condition_kwargs: dict):
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        sigma = self.float_to_tensor(sigma, batch_size, device)
        net_out = self.score_model(
            r_noisy=self.noised_coords_in_network(noisy_atom_coords, sigma),
            times=self.sigma_in_network(sigma),
            **network_condition_kwargs,
        )
        return net_out

    def predict_score(self, noisy_atom_coords, sigma, velocity):
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        padded_sigma = self.pad_sigma(sigma, batch_size, device)

        score = (
            self.a_sigma(padded_sigma) * velocity - self.c_sigma(padded_sigma) * noisy_atom_coords
        ) / (-self.b_sigma(padded_sigma))  # ! 1/t is unstable
        return score
