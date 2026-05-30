from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .assets.warpauv import WARPAUV_CFG


@configclass
class WarpAUVEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    debug_vis = True

    observation_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64)
    action_space: gym.spaces.Space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float64)
    state_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64)

    decimation = 2
    cap_episode_length = True
    episode_length_s = 6.0
    episode_length_before_reset = None
    num_actions = 8
    num_observations = 16
    num_states = 0
    use_boundaries = True
    max_auv_x = 7.0
    max_auv_y = 7.0
    max_auv_z = 1.5
    starting_depth = 8.0
    min_goal_steps = 10
    goal_completion_radius = 0.15
    goal_z_radius = 0.2
    success_speed_threshold = 0.15
    goal_dims = 3
    goal_max_distance = 4.0
    eval_mode = False

    goal_spawn_radius = 1.5
    init_xy_noise_radius = 0.35
    init_height_above_goal_range = [0.3, 1.0]
    init_guidance_rate = 0.1
    init_vel_max = 1.0
    target_heading = [1.0, 0.0, 0.0]

    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 25.0
    rew_scale_pos = 1.0
    rew_scale_ang = 0.0
    rew_scale_vel = 0.0
    rew_scale_ang_vel = 0.0
    rew_scale_lin_vel = 0.0
    rew_scale_actions = 0.05
    rew_scale_z = 0.0
    rew_scale_smooth = 0.15
    rew_scale_upright = 1.0
    reward_distance_scale = 1.5

    com_to_cob_offset = [0.0, 0.0, 0.01]
    water_rho = 997.0
    water_beta = 0.001306
    rotor_constant = 0.1 / 100.0            # 转速 --> 推力转换系数
    dyn_time_constant = 0.05
    volume = 13.0 / 997.0
    mass = 13.0

    thruster_prim_names = [
        "Thruster_R_FORWARD_FRONT",
        "Thruster_L_FORWARD_FRONT",
        "Thruster_R_FORWARD_REAR",
        "Thruster_L_FORWARD_REAR",
        "Thruster_R_UP_FRONT",
        "Thruster_L_UP_FRONT",
        "Thruster_R_UP_REAR",
        "Thruster_L_UP_REAR",
    ]

    class domain_randomization:
        use_custom_randomization = True
        com_to_cob_offset_radius = 0.05
        volume_range = [0.01134, 0.01474]
        mass_range = [13.0, 13.0]
