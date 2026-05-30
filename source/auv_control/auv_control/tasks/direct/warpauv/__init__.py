# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .warpauv_env import WarpAUVEnv
from .warpauv_env_cfg import WarpAUVEnvCfg


gym.register(
    id="Isaac-WarpAUV-Direct-v1",
    entry_point=f"{__name__}:WarpAUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WarpAUVEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.WarpAUVPPORunnerCfg,
    },
)
