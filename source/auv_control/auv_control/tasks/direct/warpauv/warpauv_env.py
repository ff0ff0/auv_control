from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import (
    BLUE_ARROW_X_MARKER_CFG,
    CUBOID_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
    VisualizationMarkers,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate

from .assets.warpauv import WARPAUV_CFG
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import ConversionFunctionBasic, DynamicsFirstOrder, get_thruster_com_and_orientations


class WarpAUVEnvWindow(BaseEnvWindow):
    """Window manager for the WarpAUV environment."""

    def __init__(self, env: WarpAUVEnv, window_name: str = "IsaacLab"):
        """Initialize the debug visualization window."""
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # Add the command manager visualization widget.
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class WarpAUVEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = WarpAUVEnvWindow

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)
    # Robot.
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Scene.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    debug_vis = True

    observation_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
    action_space: gym.spaces.Space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64)
    state_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
    # Environment.
    decimation = 2
    cap_episode_length = True
    episode_length_s = 3.0
    episode_length_before_reset = None
    num_actions = 6
    num_observations = 17
    num_states = 0
    use_boundaries = True
    max_auv_x = 7.0
    max_auv_y = 7.0
    max_auv_z = 7.0
    starting_depth = 8.0
    min_goal_steps = 100
    goal_completion_radius = 0.01
    goal_dims = 4
    eval_mode = False

    goal_spawn_radius = 2.0
    init_guidance_rate = 0.1
    init_vel_max = 1.0

    # Rewards.
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 1000.0
    rew_scale_pos = 0.2
    rew_scale_ang = 0.5
    rew_scale_vel = 0.0
    rew_scale_ang_vel = 0.0
    rew_scale_lin_vel = 0.0
    rew_scale_actions = 0.2

    # Dynamics.
    # Add this XYZ offset to COM to obtain the center of buoyancy location.
    com_to_cob_offset = [0.0, 0.0, 0.01]
    # kg/m^3
    water_rho = 997.0
    # Pa s, dynamic viscosity of water @ 50 deg F
    water_beta = 0.001306
    # Rotor constant used in Gazebo. The original source divides by 10 because
    # 0.04 was treated as roughly 10x larger than desired.
    rotor_constant = 0.1 / 100.0
    # Time constant for first-order motor dynamics.
    dyn_time_constant = 0.05
    # Assumed cubic meters, chosen to be near neutrally buoyant.
    volume = 0.022747843530591776
    # kg
    mass = 2.2701e01

    class domain_randomization:
        use_custom_randomization = True
        # Uniformly sample an offset from a sphere around the nominal COM->COB.
        com_to_cob_offset_radius = 0.05
        # Uniform lower/upper bounds for displaced volume.
        volume_range = [0.019747843530591773, 0.02574784353059178]
        mass_range = [2.2701e01, 2.2701e01]


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False
        # Initialize buffers.
        self._actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins  # Used by debug visualization.
        self._step_count = 0

        # Get thruster configurations.
        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device)
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1)

        torch.manual_seed(0)
        if self.cfg.eval_mode:
            torch.manual_seed(0)

        # Debug visualization.
        self.set_debug_vis(self.cfg.debug_vis)
        # Get specific information about the AUV.
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # TODO: pull inertias from the model or PhysX view instead of approximating.
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)
        # Estimated inertias from a solid rectangular prism model with approximate
        # side lengths 0.7m, 0.4m, and 0.2m.
        # Fake inertial values for WarpAUV based on
        # I_ii = (1/12) * mass * (len_j**2 + len_k**2).
        self.inertia_tensors[:, 0] = 0.37
        self.inertia_tensors[:, 1] = 0.97
        self.inertia_tensors[:, 2] = 1.19

        if self.cfg.mass:
            self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        else:
            self.masses = self._robot.root_physx_view._masses

        # TODO: handle these config conversions more cleanly.
        if not isinstance(self.cfg.com_to_cob_offset, torch.Tensor):
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.clone()

        if not isinstance(self.cfg.volume, torch.Tensor):
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.clone()

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True)
        # Initialize dynamics calculators.
        self._init_thruster_dynamics()
        # Set initial goals.
        self._reset_idx(self._robot._ALL_INDICES)

    def _init_thruster_dynamics(self):
        if not isinstance(self.cfg.com_to_cob_offset, torch.Tensor):
            self.cfg.com_to_cob_offset = (
                torch.tensor(self.cfg.com_to_cob_offset, device=self.device, dtype=torch.float32, requires_grad=False)
                .reshape(1, 3)
                .repeat(self.num_envs, 1)
            )

        # Create the force calculation helpers and rotor dynamics models.
        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(self.num_envs, 6, self.cfg.dyn_time_constant, self.device)
        self.thruster_conversion = ConversionFunctionBasic(self.cfg.rotor_constant)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions[:] = torch.clip(actions, -1, 1).to(self.device)

    def _apply_action(self) -> None:
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(self._actions)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        # Express the displacement from the environment origin in body coordinates.
        offset_from_origin_b = quat_apply(
            quat_conjugate(self._robot.data.root_quat_w),
            self._default_env_origins - self._robot.data.root_pos_w,
        )
        obs = torch.cat(
            [
                self._goal,
                offset_from_origin_b,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        offsets_from_origin = quat_apply(
            quat_conjugate(self._robot.data.root_quat_w),
            self._default_env_origins - self._robot.data.root_pos_w,
        )
        return _compute_rewards(
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_actions,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self.reset_terminated,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._goal,
            offsets_from_origin,
            self._completed_envs,
            self._actions,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._step_count += 1

        if self.cfg.episode_length_before_reset and self._step_count == self.cfg.episode_length_before_reset:
            time_out = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x)
                | (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y)
                | (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]
        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]

        if not self.cfg.eval_mode:
            # Randomize initial position relative to the origin.
            self._default_root_state[env_ids, :3] += self._sample_from_sphere(len(env_ids), self.cfg.goal_spawn_radius)

        self._step_count = 0
        # Apply domain randomization.
        self._reset_domain(env_ids)
        # Reset goals.
        self._reset_goal(env_ids)

        if not self.cfg.eval_mode:
            # Apply guidance by snapping a subset of envs to the target pose.
            envs_to_guide = math_utils.sample_uniform(0, 1, len(env_ids), self.device) < self.cfg.init_guidance_rate
            env_ids_to_guide = env_ids[envs_to_guide]
            self._default_root_state[env_ids_to_guide, :3] = self._default_env_origins[env_ids_to_guide, :3]
            self._default_root_state[env_ids_to_guide, 3:7] = self._goal[env_ids_to_guide, 0:4]

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)

    def _reset_goal(self, env_ids: Sequence[int]):
        # Sample a random full orientation target.
        self._goal[env_ids, 0:4] = math_utils.random_orientation(len(env_ids), device=self.device)

    def _reset_domain(self, env_ids: Sequence[int]):
        if self.cfg.domain_randomization.use_custom_randomization:
            # Randomize COM to COB offset.
            self.com_to_cob_offsets[env_ids] = (
                self.cfg.com_to_cob_offset[env_ids]
                + self._sample_from_sphere(len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius)
            )
            # Randomize displaced volume.
            vol_lower, vol_upper = self.cfg.domain_randomization.volume_range
            self.volumes[env_ids] = math_utils.sample_uniform(vol_lower, vol_upper, self.volumes[env_ids].shape, self.device)

    def _sample_from_sphere(self, num_env_ids: int, radius: float):
        coords = torch.randn((num_env_ids, 3), device=self.device)
        coords /= torch.norm(coords, dim=1, keepdim=True)
        radii = radius * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1 / 3)
        return radii * coords

    def _compute_dynamics(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute net force/torque from actions.

        Actions are in [-1, 1] and represent PWM-like commands, where -1 is full
        reverse thrust and 1 is full forward thrust.
        """
        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        motor_values = torch.clone(actions)

        # Convert PWM commands to rad/s using the mapping from the original
        # WarpAUV simulation interface.
        motor_values[torch.abs(motor_values) < 0.08] = 0.0
        motor_values[motor_values >= 0.08] = (
            -139.0 * torch.pow(motor_values[motor_values >= 0.08], 2.0)
            + 500.0 * motor_values[motor_values >= 0.08]
            + 8.28
        )
        motor_values[motor_values <= -0.08] = (
            161.0 * torch.pow(motor_values[motor_values <= -0.08], 2.0)
            + 517.86 * motor_values[motor_values <= -0.08]
            - 5.72
        )

        # Apply thruster dynamics to obtain the current motor velocities.
        # TODO: double-check that the sim dt usage here matches the original.
        motor_values = self.thruster_dynamics.update(motor_values, self.episode_length_buf * self.sim.cfg.dt)
        # Convert motor speed into thrust magnitude.
        motor_values = self.thruster_conversion.convert(motor_values)

        # Start with forces aligned with +X, then rotate into each thruster frame.
        thruster_forces[..., 0] = 1.0
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces)
        # Broadcast force magnitudes over the per-thruster force directions.
        thruster_forces = thruster_forces * motor_values.unsqueeze(-1)
        # Torque from each thruster is T = r x F.
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        # Sum over thrusters to obtain a single wrench per robot.
        thruster_forces = torch.sum(thruster_forces, dim=-2)
        thruster_torques = torch.sum(thruster_torques, dim=-2)

        # Calculate hydrodynamic contributions.
        buoyancy_forces, buoyancy_torques = self.force_calculation_functions.calculate_buoyancy_forces(
            self._robot.data.root_quat_w,
            self.cfg.water_rho,
            self.volumes,
            abs(self._gravity_magnitude),
            self.com_to_cob_offsets,
        )
        density_forces, density_torques, viscosity_forces, viscosity_torques = (
            self.force_calculation_functions.calculate_density_and_viscosity_forces(
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_w,
                self.inertia_tensors,
                self.inertia_tensors_mean,
                self.cfg.water_beta,
                self.cfg.water_rho,
                self.masses,
            )
        )

        forces = density_forces + buoyancy_forces + viscosity_forces + thruster_forces
        torques = density_torques + buoyancy_torques + viscosity_torques + thruster_torques
        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # Goal pose marker.
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_ang_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1.0)
                self.goal_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_z_ang_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_z_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1.0)
                self.goal_z_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "x_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1.0)
                marker_cfg.prim_path = "/Visuals/Command/x_b"
                self.x_b_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "z_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1.0)
                marker_cfg.prim_path = "/Visuals/Command/z_b"
                self.z_b_visualizer = VisualizationMarkers(marker_cfg)

            # Set all markers visible.
            self.goal_pos_visualizer.set_visibility(True)
            self.goal_ang_visualizer.set_visibility(True)
            self.goal_z_ang_visualizer.set_visibility(True)
            self.x_b_visualizer.set_visibility(True)
            self.z_b_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "goal_ang_visualizer"):
                self.goal_ang_visualizer.set_visibility(False)
            if hasattr(self, "goal_z_ang_visualizer"):
                self.goal_z_ang_visualizer.set_visibility(False)
            if hasattr(self, "x_b_visualizer"):
                self.x_b_visualizer.set_visibility(False)
            if hasattr(self, "z_b_visualizer"):
                self.z_b_visualizer.set_visibility(False)

    def _rotate_quat_by_euler_xyz(
        self,
        quat: torch.Tensor,
        x: float | torch.Tensor,
        y: float | torch.Tensor,
        z: float | torch.Tensor,
        device=None,
    ):
        # Assumes quat has shape [num_envs, 4].
        num_envs = quat.shape[0]
        if device is None:
            device = self.device
        if isinstance(x, float):
            x = torch.zeros(num_envs, device=device) + x
        if isinstance(y, float):
            y = torch.zeros(num_envs, device=device) + y
        if isinstance(z, float):
            z = torch.zeros(num_envs, device=device) + z

        inc_quat = math_utils.quat_from_euler_xyz(x, y, z)
        return math_utils.quat_mul(quat, inc_quat)

    def _debug_vis_callback(self, event):
        del event
        # Visualize goal positions.
        self.goal_pos_visualizer.visualize(translations=self._goal_pos_w)

        marker_scales = torch.tensor([1.0, 1.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        marker_scales[:, 0] = 1.0
        # Visualize goal orientations.
        self.goal_ang_visualizer.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=self._goal,
            scales=marker_scales,
        )

        # Visualize goal orientation through the body Z-axis as well.
        goal_z_quat = self._rotate_quat_by_euler_xyz(self._goal, 0.0, -torch.pi / 2, 0.0)
        self.goal_z_ang_visualizer.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=goal_z_quat,
            scales=marker_scales,
        )

        self.x_b_visualizer.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=self._robot.data.root_quat_w,
            scales=marker_scales,
        )

        # Visualize current body Z-axis.
        z_w_quat = self._rotate_quat_by_euler_xyz(self._robot.data.root_quat_w, 0.0, -torch.pi / 2, 0.0)
        self.z_b_visualizer.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=z_w_quat,
            scales=marker_scales,
        )


@torch.jit.script
def _compute_rewards(
    rew_scale_pos: float,
    rew_scale_ang: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_actions: float,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    goal: torch.Tensor,
    offsets_from_origin: torch.Tensor,
    completed_envs: torch.Tensor,
    actions: torch.Tensor,
):
    del rew_scale_lin_vel, lin_vel, reset_terminated, root_pos, completed_envs
    # Reward position accuracy.
    rew_pos = rew_scale_pos * torch.exp(-torch.norm(offsets_from_origin, dim=1) ** 2)
    # Reward angular accuracy.
    rew_ang = rew_scale_ang * torch.exp(-math_utils.quat_error_magnitude(goal[:, :], root_quat[:, :]))
    # Penalize angular velocity.
    rew_ang_vel = rew_scale_ang_vel * torch.exp(-torch.norm(ang_vel, dim=1) ** 2)
    # Penalize energy consumption.
    rew_action = rew_scale_actions * torch.exp(-torch.norm(actions, dim=1) ** 2)
    return rew_ang + rew_action + rew_pos + rew_ang_vel
