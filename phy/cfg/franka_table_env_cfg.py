"""Configuration for the Franka table environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from phy.utils.assets import DEFAULT_USD_ROOT_GRASP

from .franka_base_env_cfg import FrankaBaseEnvCfg


@configclass
class FrankaTableEnvCfg(FrankaBaseEnvCfg):
    """Franka table task configuration."""

    table_size = (0.80, 0.80, 0.05)
    table_center = (0.55, 0.0, -0.025) # table surface height will be 0.0
    object_xy_offset = (0.0, 0.0)
    object_table_clearance = 0.005
    default_object_height = 0.20
    override_object_physics = False
    usd_root: str = str(DEFAULT_USD_ROOT_GRASP)

    num_grasps = 0 # 0 means load all available grasps for the object
    start_object_idx = 0
    require_grasps = False
    asset_axis_convention = "y_up_to_z_up" # default for THOR assets (z is still defined as up axis, but the semantic up direction is along the y axis in THOR assets, so we need to rotate to z up for Isaac Lab)

    enable_recording_camera = False
    recording_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/RecordCamera",
        update_period=0.0,
        height=960,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.0, 0.0, 1.0),
            rot=(0.35355, -0.61237, -0.61237, 0.35355),
            convention="ros",
        ),
    )
