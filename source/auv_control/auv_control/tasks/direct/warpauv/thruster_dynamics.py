"""
Thruster dynamics and model for WarpAUV.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from isaaclab.utils.math import quat_from_euler_xyz


def get_thruster_com_and_orientations(device: torch.device):
    """
    Return thruster extrinsics for a single vehicle.

    TODO: This would be cleaner if it came directly from the USD/URDF model
    through named actuators instead of being hard-coded here.
    """
    def create_tf_rpy(x, y, z, rr, rp, ry):
        shift = torch.tensor([x, y, z], dtype=torch.float32)
        quat = quat_from_euler_xyz(
            torch.tensor([rr], dtype=torch.float32),
            torch.tensor([rp], dtype=torch.float32),
            torch.tensor([ry], dtype=torch.float32),
        )[0]
        return shift, quat

    def create_tf_quat(x, y, z, w, vx, vy, vz):
        shift = torch.tensor([x, y, z], dtype=torch.float32)
        quat = torch.tensor([w, vx, vy, vz], dtype=torch.float32)
        return shift, quat

    thruster_info = dict(
        drive_left=create_tf_quat(-0.4127, 0.1506, -0.0889, 1.0, 0.0, 0.0, 0.0),
        drive_right=create_tf_quat(-0.4127, -0.1506, -0.0889, 1.0, 0.0, 0.0, 0.0),
        rear_left=create_tf_rpy(-0.3030, 0.1461, -0.1587, 0.0, -0.785398, 1.5708),
        rear_right=create_tf_rpy(-0.3030, -0.1461, -0.1587, 0.0, -0.785398, -1.5708),
        front_right=create_tf_rpy(0.0585, -0.1461, -0.0540, 0.0, 0.785398, -1.5708),
        front_left=create_tf_rpy(0.0585, 0.1461, -0.0540, 0.0, 0.785398, 1.5708),
    )

    # Vector pointing from COM to thruster location, shape: (thruster, 3).
    # Thruster ordering:
    # 0 - drive_left
    # 1 - drive_right
    # 2 - rear_left
    # 3 - rear_right
    # 4 - front_left
    # 5 - front_right
    thruster_com_offsets = torch.stack(
        [
            thruster_info["drive_left"][0],
            thruster_info["drive_right"][0],
            thruster_info["rear_left"][0],
            thruster_info["rear_right"][0],
            thruster_info["front_left"][0],
            thruster_info["front_right"][0],
        ],
        dim=0,
    ).to(device)
    # Quaternion mapping COM frame -> thruster frame, shape: (thruster, 4).
    thruster_quats = torch.stack(
        [
            thruster_info["drive_left"][1],
            thruster_info["drive_right"][1],
            thruster_info["rear_left"][1],
            thruster_info["rear_right"][1],
            thruster_info["front_left"][1],
            thruster_info["front_right"][1],
        ],
        dim=0,
    ).to(device)
    return thruster_com_offsets, thruster_quats


class Dynamics(ABC):
    def __init__(self, num_envs: int, num_thrusters_per_env: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.num_thrusters_per_env = num_thrusters_per_env
        self.device = device
        self.reset_all()

    def reset(self, mask_arr: torch.Tensor):
        # mask_arr is a boolean tensor of shape (num_envs,) marking which
        # environments should reset their thruster state.
        self.state[mask_arr, :] = 0.0
        self.prev_time[mask_arr] = -1.0

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
        dt = t - self.prev_time
        alpha = torch.exp(-dt / self.tau)
        # NOTE: This follows the original source exactly: alpha is zeroed out so the
        # state snaps directly to the command instead of applying first-order lag.
        alpha = torch.zeros_like(alpha)
        self.state = self.state * alpha.unsqueeze(-1) + (1.0 - alpha).unsqueeze(-1) * cmd
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
