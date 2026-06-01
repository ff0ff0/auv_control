from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import (
    CUBOID_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    VisualizationMarkers,
)
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate

from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import (
    ConversionFunctionBasic,
    ConversionFunctionT200,
    DynamicsFirstOrder,
    DynamicsT200,
    get_thruster_com_and_orientations,
)
from .warpauv_env_cfg import WarpAUVEnvCfg


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


WarpAUVEnvCfg.ui_window_class_type = WarpAUVEnvWindow


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False
        # Initialize buffers.
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._target_heading = torch.tensor(self.cfg.target_heading, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins  # Used by debug visualization.
        self._step_count = 0
        self._flow_vels_w = torch.tensor(self.cfg.flow_velocity, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )

        # Get thruster configurations.
        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(
            self.device,
            self.cfg.robot_cfg.spawn.usd_path,
            self.cfg.thruster_prim_names,
        )
        self.num_thrusters = self.thruster_com_offsets.shape[0]
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1)

        torch.manual_seed(0)
        if self.cfg.eval_mode:
            torch.manual_seed(0)

        # Debug visualization.
        self.set_debug_vis(self.cfg.debug_vis)
        # Get specific information about the AUV.
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # TODO: pull inertias from the model or PhysX view directly.
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)
        # Principal inertias synchronized to the BlueROV Heavy URDF base_link:
        # ixx=0.26, iyy=0.23, izz=0.37.
        self.inertia_tensors[:, 0] = 0.26
        self.inertia_tensors[:, 1] = 0.23
        self.inertia_tensors[:, 2] = 0.37

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
        self.force_calculation_functions = HydrodynamicForceModels(
            self.num_envs,
            self.device,
            self.step_dt,
            torch.tensor(self.cfg.added_mass_diag, device=self.device, dtype=torch.float32),
            torch.tensor(self.cfg.linear_damping_diag, device=self.device, dtype=torch.float32),
            torch.tensor(self.cfg.quadratic_damping_diag, device=self.device, dtype=torch.float32),
            self.cfg.hydro_accel_filter_alpha,
            False,
        )
        if self.cfg.use_marinegym_t200_model:
            self.thruster_dynamics = DynamicsT200(
                self.num_envs,
                self.num_thrusters,
                self.cfg.t200_tau_up,
                self.cfg.t200_tau_down,
                self.cfg.t200_time_constant,
                self.cfg.t200_deadzone,
                self.cfg.t200_forward_gain,
                self.cfg.t200_forward_offset,
                self.cfg.t200_reverse_gain,
                self.cfg.t200_reverse_offset,
                self.device,
            )
            self.thruster_conversion = ConversionFunctionT200(self.cfg.t200_force_constant_scale)
        else:
            self.thruster_dynamics = DynamicsFirstOrder(
                self.num_envs, self.num_thrusters, self.cfg.dyn_time_constant, self.device
            )
            self.thruster_conversion = ConversionFunctionBasic(self.cfg.rotor_constant)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions[:] = self._actions
        self._actions[:] = torch.clip(actions, -1, 1).to(self.device)

    # 执行动作
    def _apply_action(self) -> None:
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(self._actions)        # 根据定义的运动学方程计算力和力矩
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        target_delta = self._goal - self._robot.data.root_pos_w
        heading = quat_apply(
            self._robot.data.root_quat_w,
            torch.tensor([[1.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1),
        )
        relative_heading = self._target_heading - heading
        obs = torch.cat(
            [
                target_delta,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                relative_heading,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        target_delta = self._goal - self._robot.data.root_pos_w
        heading = quat_apply(
            self._robot.data.root_quat_w,
            torch.tensor([[1.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1),
        )
        relative_heading = self._target_heading - heading
        body_up = quat_apply(
            self._robot.data.root_quat_w,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1),
        )
        return _compute_rewards(
            self.cfg.rew_scale_completion,
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_actions,
            self.cfg.rew_scale_z,
            self.cfg.rew_scale_smooth,
            self.cfg.rew_scale_upright,
            self.cfg.reward_distance_scale,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            target_delta,
            relative_heading,
            body_up[:, 2],
            self._actions,
            self._previous_actions,
            self._completed_envs,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._step_count += 1

        if self.cfg.episode_length_before_reset and self._step_count == self.cfg.episode_length_before_reset:
            time_out = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        target_delta = self._goal - self._robot.data.root_pos_w
        xy_error = torch.norm(target_delta[:, :2], dim=1)
        z_error = torch.abs(target_delta[:, 2])
        speed = torch.norm(self._robot.data.root_lin_vel_b, dim=1)
        success_now = (
            (xy_error < self.cfg.goal_completion_radius)
            & (z_error < self.cfg.goal_z_radius)
            & (speed < self.cfg.success_speed_threshold)
        )
        self._completion_buffer[success_now] += 1
        self._completion_buffer[~success_now] = 0
        self._completed_envs[:] = self._completion_buffer >= self.cfg.min_goal_steps
        distance = torch.norm(target_delta, dim=1)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x)
                | (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y)
                | (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        misbehave = (self._robot.data.root_pos_w[:, 2] < 0.2) | (distance > self.cfg.goal_max_distance)
        has_nan = torch.isnan(self._robot.data.root_pos_w).any(dim=1) | torch.isnan(self._robot.data.root_lin_vel_b).any(dim=1)

        return out_of_bounds | self._completed_envs | misbehave | has_nan, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]
        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]
        self._default_root_state[env_ids, 2] = self.cfg.starting_depth
        self._default_root_state[env_ids, 7:13] = 0.0
        self._completion_buffer[env_ids] = 0.0
        self._completed_envs[env_ids] = False
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.force_calculation_functions.reset(env_ids)
        self.thruster_dynamics.reset(env_ids)

        # Reset goals first so the initial state can be sampled above them.
        self._reset_goal(env_ids)

        if not self.cfg.eval_mode:
            # Initialize near the target point in xy, but above it in z.
            self._default_root_state[env_ids, :2] = (
                self._goal[env_ids, :2]
                + self._sample_from_circle(len(env_ids), self.cfg.init_xy_noise_radius)
            )
            self._default_root_state[env_ids, 2] = self._goal[env_ids, 2] + math_utils.sample_uniform(
                self.cfg.init_height_above_goal_range[0],
                self.cfg.init_height_above_goal_range[1],
                (len(env_ids),),
                self.device,
            )

        self._step_count = 0
        # Apply domain randomization.
        self._reset_domain(env_ids)

        if not self.cfg.eval_mode:
            # Apply guidance by snapping a subset of envs even closer above the target point.
            envs_to_guide = math_utils.sample_uniform(0, 1, len(env_ids), self.device) < self.cfg.init_guidance_rate
            env_ids_to_guide = env_ids[envs_to_guide]
            self._default_root_state[env_ids_to_guide, :2] = self._goal[env_ids_to_guide, :2]
            self._default_root_state[env_ids_to_guide, 2] = self._goal[env_ids_to_guide, 2] + 0.2

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)

    def _reset_goal(self, env_ids: Sequence[int]):
        planar_offsets = self._sample_from_circle(len(env_ids), self.cfg.goal_spawn_radius)
        self._goal[env_ids, :2] = self._default_env_origins[env_ids, :2] + planar_offsets
        self._goal[env_ids, 2] = self.cfg.starting_depth
        self._goal_pos_w[env_ids] = self._goal[env_ids]

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

    def _sample_from_circle(self, num_env_ids: int, radius: float):
        sampled_radius = radius * torch.sqrt(torch.rand((num_env_ids, 1), device=self.device))
        sampled_theta = torch.rand((num_env_ids, 1), device=self.device) * 2 * torch.pi
        sampled_x = sampled_radius * torch.cos(sampled_theta)
        sampled_y = sampled_radius * torch.sin(sampled_theta)
        return torch.cat([sampled_x, sampled_y], dim=1)

    # 将各个螺旋桨输入的PWM，转换为对应的力和力矩输出
    def _compute_dynamics(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute net force/torque from actions.

        Actions are in [-1, 1] and represent PWM-like commands, where -1 is full
        reverse thrust and 1 is full forward thrust.
        """
        thruster_forces = torch.zeros((self.num_envs, self.num_thrusters, 3), device=self.device, dtype=torch.float)
        motor_values = torch.clone(actions)

        if not self.cfg.use_marinegym_t200_model:
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
        hydro_forces, hydro_torques = self.force_calculation_functions.calculate_hydrodynamic_wrench(
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._flow_vels_w,
            self._robot.data.root_quat_w,
        )

        forces = hydro_forces + buoyancy_forces + thruster_forces
        torques = hydro_torques + buoyancy_torques + thruster_torques
        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # Goal pose marker.
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

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
            self.x_b_visualizer.set_visibility(True)
            self.z_b_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
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
    rew_scale_completion: float,
    rew_scale_pos: float,
    rew_scale_ang: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_actions: float,
    rew_scale_z: float,
    rew_scale_smooth: float,
    rew_scale_upright: float,
    reward_distance_scale: float,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    goal_delta: torch.Tensor,
    relative_heading: torch.Tensor,
    uprightness: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    completed_envs: torch.Tensor,
):
    z_error = torch.abs(goal_delta[:, 2])
    distance = torch.norm(torch.cat([goal_delta, relative_heading], dim=1), dim=1)
    spin = torch.square(ang_vel[:, 2])
    action_delta = torch.norm(actions - previous_actions, dim=1)

    reward_pose = 0.5 * rew_scale_pos / (1.0 + torch.square(reward_distance_scale * distance))
    rew_z = rew_scale_z * torch.exp(-torch.square(2.0 * z_error))
    rew_vel = rew_scale_lin_vel * torch.exp(-torch.square(torch.norm(lin_vel, dim=1)))
    rew_upright = rew_scale_upright * torch.square(torch.clamp((uprightness + 1.0) / 2.0, min=0.0, max=1.0))
    rew_ang_vel = rew_scale_ang_vel * torch.exp(-torch.square(torch.norm(ang_vel, dim=1)))
    rew_spin = 1.0 / (1.0 + torch.square(spin))
    rew_action = rew_scale_actions * torch.exp(-torch.norm(actions, dim=1))
    rew_smooth = rew_scale_smooth * torch.exp(-action_delta)
    rew_heading = rew_scale_ang * torch.exp(-torch.norm(relative_heading, dim=1))
    rew_completion = rew_scale_completion * completed_envs.float()

    return (
        reward_pose
        + reward_pose * (rew_upright + rew_spin + rew_heading + rew_z + rew_vel + rew_ang_vel)
        + rew_action
        + rew_smooth
        + rew_completion
    )
