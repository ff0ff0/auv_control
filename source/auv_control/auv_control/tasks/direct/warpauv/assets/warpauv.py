import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


USD_PATH = "/home/linc/project/RL/isaac_underwater/isaac_navigation_task/assets/BlueRov/bluerov_heavy/bluerov_heavy/bluerov_heavy.usd"

WARPAUV_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        copy_from_source=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5.0),
    ),
)
"""Configuration for the WarpAUV rigid body asset."""
