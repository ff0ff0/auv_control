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
    max_auv_x = 10.0
    max_auv_y = 10.0
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

    com_to_cob_offset = [0.0, 0.0, 0.0]     # 质心到浮心的偏移，用于计算浮力矩
    water_rho = 997.0                        # 水密度，决定浮力和部分水动力尺度
    water_beta = 0.001306                    # 旧版线性黏性阻力系数，当前 MarineGym 风格水动力中基本未使用
    flow_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]          # 环境水流速度 [vx, vy, vz, wx, wy, wz]
    hydro_accel_filter_alpha = 0.3                          # 体速度差分求加速度时的一阶滤波系数
    added_mass_diag = [5.5, 12.7, 14.57, 0.12, 0.12, 0.12]  # 6 自由度附加质量对角项
    linear_damping_diag = [4.03, 6.22, 5.18, 0.07, 0.07, 0.07]          # 6 自由度线性阻尼对角项
    quadratic_damping_diag = [18.18, 21.66, 36.99, 1.55, 1.55, 1.55]    # 6 自由度二次阻尼对角项
    use_marinegym_t200_model = True          # 是否启用 MarineGym 风格 T200 推进器模型
    t200_tau_up = 0.43                       # 油门上升时的推进器内部响应系数
    t200_tau_down = 0.43                     # 油门下降时的推进器内部响应系数
    t200_time_constant = 0.01                # rpm 一阶响应时间常数
    t200_deadzone = 0.075                    # T200 推进器死区阈值
    t200_forward_gain = 3659.9               # 正向油门到目标 rpm 的线性增益
    t200_forward_offset = 345.21             # 正向油门到目标 rpm 的线性偏置
    t200_reverse_gain = 3494.4               # 反向油门到目标 rpm 的线性增益
    t200_reverse_offset = 433.50             # 反向油门到目标 rpm 的线性偏置
    t200_force_constant_scale = 1.0          # T200 经验推力曲线的整体缩放系数
    rotor_constant = 0.1 / 100.0             # 旧版简化推进器模型的转速到推力系数
    dyn_time_constant = 0.05                 # 旧版简化推进器模型的一阶动态时间常数
    volume = 12.8 / 997.0                    # 排水体积，决定浮力大小
    mass = 13.0                              # 机器人质量

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
        mass_range = [12.5, 13.0]
