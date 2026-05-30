"""
Compute hydrodynamic forces and torques on a rigid body.

Based on the MuJoCo hydrodynamic model:
https://mujoco.readthedocs.io/en/3.0.1/computation/fluid.html
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate


@dataclass
class HydrodynamicForceModels:
    num_envs: int
    device: torch.device
    debug: bool = False

    def calculate_buoyancy_forces(
        self,
        root_quats_w: torch.Tensor,
        fluid_density: float,
        volumes: torch.Tensor,
        g_mag: float,
        com_to_cob_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute buoyancy forces/torques for a fully submerged rigid body.

        Returned forces and torques are expressed in the body root frame.
        Gravity itself is still applied by Isaac Sim.
        """
        buoyancy_directions_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        # Buoyancy opposes gravity in the world frame.
        buoyancy_directions_w[..., 2] = 1.0
        buoyancy_directions_b = quat_apply(quat_conjugate(root_quats_w), buoyancy_directions_w)

        # TODO: Ideally we would compute the equivalent wrench at the vehicle root
        # rather than directly at the center of buoyancy.
        buoyancy_forces_at_cob_b = buoyancy_directions_b * fluid_density * volumes.repeat(1, 3) * g_mag
        buoyancy_torques_b = torch.cross(com_to_cob_offsets, buoyancy_forces_at_cob_b, dim=-1)
        return buoyancy_forces_at_cob_b, buoyancy_torques_b

    def _calculate_inferred_half_dimensions(self, inertias: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Infer an equivalent inertia-box half-extent from principal inertias."""
        return torch.sqrt(
            (3.0 / (2.0 * masses.repeat(1, 3)))
            * (torch.roll(inertias, 1, 1) + torch.roll(inertias, -1, 1) - inertias)
        )

    def calculate_quadratic_drag_forces(
        self,
        root_linvels_b: torch.Tensor,
        root_angvels_b: torch.Tensor,
        inertias: torch.Tensor,
        masses: torch.Tensor,
        fluid_density_rho: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute quadratic drag in the body frame."""
        ri = self._calculate_inferred_half_dimensions(inertias, masses)
        rj = torch.roll(ri, 1, 1)
        rk = torch.roll(ri, -1, 1)

        forces = -2.0 * fluid_density_rho * rj * rk * torch.abs(root_linvels_b) * root_linvels_b
        torques = (
            -0.5
            * fluid_density_rho
            * ri
            * (torch.pow(rj, 4) + torch.pow(rk, 4))
            * torch.abs(root_angvels_b)
            * root_angvels_b
        )
        return forces, torques

    def calculate_linear_viscous_forces(
        self,
        root_linvels_b: torch.Tensor,
        root_angvels_b: torch.Tensor,
        inertias: torch.Tensor,
        masses: torch.Tensor,
        fluid_viscosity_beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear viscous drag in the body frame."""
        ri = self._calculate_inferred_half_dimensions(inertias, masses)
        r_eq = torch.mean(ri, 1, keepdim=True).repeat(1, 3)

        forces = -6.0 * fluid_viscosity_beta * torch.pi * r_eq * root_linvels_b
        torques = -8.0 * fluid_viscosity_beta * torch.pi * torch.pow(r_eq, 3) * root_angvels_b
        return forces, torques

    def calculate_density_and_viscosity_forces(
        self,
        root_quats_w: torch.Tensor,
        root_linvels_w: torch.Tensor,
        root_angvels_w: torch.Tensor,
        inertias: torch.Tensor,
        inertias_mean: torch.Tensor,
        water_beta: float,
        water_rho: float,
        masses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del inertias_mean
        # Transform world-frame velocities into the body frame before evaluating
        # the hydrodynamic models.
        root_quats_b = quat_conjugate(root_quats_w)
        root_linvels_b = quat_apply(root_quats_b, root_linvels_w)
        root_angvels_b = quat_apply(root_quats_b, root_angvels_w)

        f_d, g_d = self.calculate_quadratic_drag_forces(root_linvels_b, root_angvels_b, inertias, masses, water_rho)
        f_v, g_v = self.calculate_linear_viscous_forces(root_linvels_b, root_angvels_b, inertias, masses, water_beta)
        return f_d, g_d, f_v, g_v
