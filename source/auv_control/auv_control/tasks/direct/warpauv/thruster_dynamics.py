"""
Thruster dynamics and model for WarpAUV.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Sequence

import torch


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def get_thruster_com_and_orientations(
    device: torch.device,
    usd_path: str,
    thruster_prim_names: Sequence[str],
):
    """
    Return thruster extrinsics for a single vehicle by reading the USD asset.

    TODO: This would be cleaner if it came directly from the USD/URDF model
    through named actuators instead of being hard-coded here.
    """
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:
        raise ImportError("Reading thruster transforms from USD requires the pxr Python package.") from exc

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"Unable to open USD asset: {usd_path}")

    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        raise ValueError(f"USD asset has no default prim: {usd_path}")
    xform_cache = UsdGeom.XformCache()

    normalized_to_prim = {}
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        normalized_to_prim[_normalize_identifier(str(prim.GetPath()))] = prim
        normalized_to_prim[_normalize_identifier(prim.GetName())] = prim

    thruster_positions = []
    thruster_quats = []

    # Vector pointing from COM to thruster location, shape: (thruster, 3).
    # Quaternion mapping COM frame -> thruster frame, shape: (thruster, 4).
    for thruster_name in thruster_prim_names:
        normalized_name = _normalize_identifier(thruster_name)
        matches = []
        for key, prim in normalized_to_prim.items():
            if normalized_name in key:
                matches.append(prim)

        unique_matches = []
        seen_paths = set()
        for prim in matches:
            path = str(prim.GetPath())
            if path not in seen_paths:
                seen_paths.add(path)
                unique_matches.append(prim)

        if len(unique_matches) != 1:
            raise ValueError(
                f"Expected exactly one USD prim match for thruster '{thruster_name}', found "
                f"{len(unique_matches)} in asset {usd_path}."
            )

        prim = unique_matches[0]
        xformable = UsdGeom.Xformable(prim)
        root_xformable = UsdGeom.Xformable(root_prim)

        if hasattr(xformable, "ComputeRelativeTransform"):
            relative_matrix, _ = xformable.ComputeRelativeTransform(root_prim)
        else:
            # Older USD Python bindings commonly expose only local-to-world
            # transforms, so compute the relative transform manually.
            prim_world = xform_cache.GetLocalToWorldTransform(prim)
            root_world = xform_cache.GetLocalToWorldTransform(root_prim)
            relative_matrix = prim_world * root_world.GetInverse()

            # If the root itself has no authored xform ops, the direct local
            # transform on the thruster prim is often already the desired value.
            if not root_xformable:
                local_matrix, _ = xformable.GetLocalTransformation()
                relative_matrix = local_matrix

        transform = Gf.Transform(relative_matrix)
        translation = transform.GetTranslation()
        rotation = transform.GetRotation().GetQuat()
        imag = rotation.GetImaginary()

        thruster_positions.append(
            torch.tensor([translation[0], translation[1], translation[2]], dtype=torch.float32, device=device)
        )
        thruster_quats.append(
            torch.tensor([rotation.GetReal(), imag[0], imag[1], imag[2]], dtype=torch.float32, device=device)
        )

    thruster_com_offsets = torch.stack(thruster_positions, dim=0).to(device)
    thruster_quats = torch.stack(thruster_quats, dim=0).to(device)
    return thruster_com_offsets, thruster_quats


class Dynamics(ABC):
    def __init__(self, num_envs: int, num_thrusters_per_env: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.num_thrusters_per_env = num_thrusters_per_env
        self.device = device
        self.reset_all()

    def reset(self, env_ids: torch.Tensor):
        self.state[env_ids, :] = 0.0
        self.prev_time[env_ids] = -1.0

    def reset_all(self):
        self.state = torch.zeros(
            (self.num_envs, self.num_thrusters_per_env),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.prev_time = torch.full(
            (self.num_envs,),
            -1.0,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

    @abstractmethod
    def update(self, cmd: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DynamicsFirstOrder(Dynamics):
    def __init__(self, num_envs: int, num_thrusters_per_env: int, tau: float, device: torch.device):
        super().__init__(num_envs=num_envs, num_thrusters_per_env=num_thrusters_per_env, device=device)
        self.tau = tau

    def update(self, cmd: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Set previously uninitialized times to the current time in those envs.
        self.prev_time[self.prev_time < 0] = t[self.prev_time < 0]
        # Because dt = 0 for freshly initialized envs, alpha would be 1.0 and the
        # state would remain unchanged on the first step.
        dt = torch.clamp(t - self.prev_time, min=0.0)
        alpha = torch.exp(-dt / self.tau)
        self.state = self.state * alpha.unsqueeze(-1) + (1.0 - alpha).unsqueeze(-1) * cmd
        self.prev_time = t
        return self.state


class DynamicsT200(Dynamics):
    def __init__(
        self,
        num_envs: int,
        num_thrusters_per_env: int,
        tau_up: float,
        tau_down: float,
        time_constant: float,
        deadzone: float,
        forward_gain: float,
        forward_offset: float,
        reverse_gain: float,
        reverse_offset: float,
        device: torch.device,
    ):
        self.tau_up = tau_up
        self.tau_down = tau_down
        self.time_constant = time_constant
        self.deadzone = deadzone
        self.forward_gain = forward_gain
        self.forward_offset = forward_offset
        self.reverse_gain = reverse_gain
        self.reverse_offset = reverse_offset
        super().__init__(num_envs=num_envs, num_thrusters_per_env=num_thrusters_per_env, device=device)

    def reset_all(self):
        super().reset_all()
        self.throttle = torch.zeros(
            (self.num_envs, self.num_thrusters_per_env),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self.throttle[env_ids, :] = 0.0

    def update(self, cmd: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.prev_time[self.prev_time < 0] = t[self.prev_time < 0]
        tau = torch.where(cmd > self.throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(torch.as_tensor(tau, device=self.device, dtype=torch.float32), 0.0, 1.0)
        self.throttle = self.throttle + tau * (cmd - self.throttle)

        target_rpm = torch.where(
            self.throttle > self.deadzone,
            self.forward_gain * self.throttle + self.forward_offset,
            torch.where(
                self.throttle < -self.deadzone,
                self.reverse_gain * self.throttle - self.reverse_offset,
                torch.zeros_like(self.throttle),
            ),
        )

        dt = torch.clamp(t - self.prev_time, min=0.0)
        alpha = torch.exp(-dt / self.time_constant)
        self.state = self.state * alpha.unsqueeze(-1) + (1.0 - alpha).unsqueeze(-1) * target_rpm
        self.prev_time = t
        return self.state


@dataclass
class ConversionFunction(ABC):
    @abstractmethod
    def convert(self, cmd: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConversionFunctionBasic(ConversionFunction):
    rotor_constant: float

    def __init__(self, rotor_constant: float):
        super().__init__()
        self.rotor_constant = rotor_constant

    def convert(self, cmd: torch.Tensor) -> torch.Tensor:
        # Convert rotor angular velocity command into thrust.
        return self.rotor_constant * torch.abs(cmd) * cmd


class ConversionFunctionT200(ConversionFunction):
    force_constant_scale: float

    def __init__(self, force_constant_scale: float = 1.0):
        super().__init__()
        self.force_constant_scale = force_constant_scale

    # 根据经验模型，将电机转速转换为推力
    def convert(self, cmd: torch.Tensor) -> torch.Tensor:
        positive_thrust = 4.7368e-07 * torch.square(cmd) - 1.9275e-04 * cmd + 8.4452e-02
        negative_thrust = -3.8442e-07 * torch.square(cmd) - 1.6186e-04 * cmd - 3.9139e-02
        thrust_kgf = torch.where(cmd > 0.0, positive_thrust, negative_thrust)
        return self.force_constant_scale * 9.81 * thrust_kgf
