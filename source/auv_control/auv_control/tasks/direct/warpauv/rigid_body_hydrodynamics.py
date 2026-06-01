"""MarineGym-style hydrodynamic wrench model for a rigid underwater vehicle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate


@dataclass
class HydrodynamicForceModels:
    num_envs: int
    device: torch.device
    dt: float
    added_mass_diag: torch.Tensor
    linear_damping_diag: torch.Tensor
    quadratic_damping_diag: torch.Tensor
    accel_filter_alpha: float = 0.3
    debug: bool = False

    def __post_init__(self):
        del self.debug
        self.added_mass_matrix = torch.diag(self.added_mass_diag).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.linear_damping_matrix = torch.diag(self.linear_damping_diag).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.quadratic_damping_matrix = torch.diag(self.quadratic_damping_diag).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.prev_body_vels = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self.prev_body_acc = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | list[int] | None = None):
        if env_ids is None:
            self.prev_body_vels.zero_()
            self.prev_body_acc.zero_()
            return
        self.prev_body_vels[env_ids] = 0.0
        self.prev_body_acc[env_ids] = 0.0

    def calculate_buoyancy_forces(
        self,
        root_quats_w: torch.Tensor,
        fluid_density: float,
        volumes: torch.Tensor,
        g_mag: float,
        com_to_cob_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute buoyancy using the existing center-of-buoyancy offset model."""
        buoyancy_directions_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        buoyancy_directions_w[..., 2] = 1.0
        buoyancy_directions_b = quat_apply(quat_conjugate(root_quats_w), buoyancy_directions_w)
        buoyancy_forces_at_cob_b = buoyancy_directions_b * fluid_density * volumes.repeat(1, 3) * g_mag
        buoyancy_torques_b = torch.cross(com_to_cob_offsets, buoyancy_forces_at_cob_b, dim=-1)
        return buoyancy_forces_at_cob_b, buoyancy_torques_b

    def calculate_hydrodynamic_wrench(
        self,
        root_linvels_b: torch.Tensor,
        root_angvels_b: torch.Tensor,
        flow_vels_w: torch.Tensor,
        root_quats_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MarineGym-style damping, added-mass, and Coriolis wrench."""
        body_vels = torch.cat([root_linvels_b, root_angvels_b], dim=-1)
        flow_linvels_b = quat_apply(quat_conjugate(root_quats_w), flow_vels_w[:, :3])
        flow_angvels_b = quat_apply(quat_conjugate(root_quats_w), flow_vels_w[:, 3:])
        flow_vels_b = torch.cat([flow_linvels_b, flow_angvels_b], dim=-1)
        rel_body_vels = body_vels - flow_vels_b

        body_acc = self._calculate_acc(rel_body_vels)
        damping = self._calculate_damping(rel_body_vels)
        added_mass = self._calculate_added_mass(body_acc)
        coriolis = self._calculate_coriolis(rel_body_vels)

        hydro = -(added_mass + coriolis + damping)
        return hydro[:, :3], hydro[:, 3:]

    def _calculate_acc(self, body_vels: torch.Tensor) -> torch.Tensor:
        acc = (body_vels - self.prev_body_vels) / self.dt
        filtered_acc = (1.0 - self.accel_filter_alpha) * self.prev_body_acc + self.accel_filter_alpha * acc
        self.prev_body_vels = body_vels.clone()
        self.prev_body_acc = filtered_acc.clone()
        return filtered_acc

    def _calculate_damping(self, body_vels: torch.Tensor) -> torch.Tensor:
        maintained_body_vels = torch.diag_embed(body_vels)
        maintained_body_vels[:, 1, 5] = body_vels[:, 5]
        maintained_body_vels[:, 2, 4] = body_vels[:, 4]
        maintained_body_vels[:, 4, 2] = body_vels[:, 2]
        maintained_body_vels[:, 5, 1] = body_vels[:, 1]
        damping_matrix = self.linear_damping_matrix + self.quadratic_damping_matrix * torch.abs(maintained_body_vels)
        return (damping_matrix @ body_vels.unsqueeze(-1)).squeeze(-1)

    def _calculate_added_mass(self, body_acc: torch.Tensor) -> torch.Tensor:
        return (self.added_mass_matrix @ body_acc.unsqueeze(-1)).squeeze(-1)

    def _calculate_coriolis(self, body_vels: torch.Tensor) -> torch.Tensor:
        ab = (self.added_mass_matrix @ body_vels.unsqueeze(-1)).squeeze(-1)
        coriolis = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        coriolis[:, :3] = -torch.cross(ab[:, :3], body_vels[:, 3:], dim=1)
        coriolis[:, 3:] = -(
            torch.cross(ab[:, :3], body_vels[:, :3], dim=1)
            + torch.cross(ab[:, 3:], body_vels[:, 3:], dim=1)
        )
        return coriolis
