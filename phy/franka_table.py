"""Franka table environment with cached object-relative grasps."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import TiledCamera

from phy.cfg.franka_table_env_cfg import FrankaTableEnvCfg
from phy.franka_base_env import FrankaBaseEnv
from phy.utils.assets import (
    discover_thor_assets,
    load_usd_asset_metadata,
    object_height,
    select_env_assets,
)
from phy.utils.transforms import axis_rotation_wxyz, transform_from_pos_wxyz


class FrankaTableEnv(FrankaBaseEnv):
    """Franka environment with a table and one THOR object per scene."""

    cfg: FrankaTableEnvCfg

    def __init__(
        self, cfg: FrankaTableEnvCfg, render_mode: str | None = None, **kwargs
    ):
        """Initialize table scenes and cache object-relative grasp transforms.

        Args:
            cfg: Environment settings; ``cfg.scene.num_envs`` is the scene count N.
            render_mode: Rendering mode forwarded to the base environment.
            **kwargs: Additional base-environment initialization arguments.

        Returns:
            None. Creates N object instances and caches grasp tensors shaped
            ``(G, 4, 4)`` per loaded asset, where G is its grasp count.
        """
        self.asset_metadata = load_usd_asset_metadata()
        self.selected_assets, self.object_grasps = select_env_assets(
            discover_thor_assets(cfg.usd_root),
            cfg.scene.num_envs,
            cfg.start_object_idx,
            cfg.num_grasps,
            cfg.require_grasps,
        )
        self.object_orientation_offset = axis_rotation_wxyz(
            cfg.asset_axis_convention
        )  # in case the asset's axis convention is different from Isaac Lab's, we add the offset to make everything consistent in Isaac Lab's convention (z is up)
        self.objects = []
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_task_scene(self) -> None:
        """Spawn one table and object per environment, plus the optional camera.

        Uses ``self.cfg`` and ``self.selected_assets``; no explicit arguments.
        Table size and position are XYZ triples; object orientation is a
        ``(4,)`` quaternion in ``(w, x, y, z)`` order.

        Returns:
            None. Registers objects and the enabled recording camera in the scene.
        """
        # use the same table config for all envs
        table_cfg = sim_utils.CuboidCfg(
            size=self.cfg.table_size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.52)
            ),
        )

        assert len(self.selected_assets) == self.cfg.scene.num_envs

        for env_id, asset in enumerate(self.selected_assets):
            env_path = self.scene.env_prim_paths[env_id]
            table_cfg.func(
                f"{env_path}/Table", table_cfg, translation=self.cfg.table_center
            )

            # spawning the object
            height = object_height(
                asset.asset_id,
                self.asset_metadata,
                self.cfg.default_object_height,
                self.object_orientation_offset,
            )
            obj_init_position = (
                self.cfg.table_center[0] + self.cfg.object_xy_offset[0],
                self.cfg.table_center[1] + self.cfg.object_xy_offset[1],
                self.cfg.table_center[2]
                + self.cfg.table_size[2] * 0.5
                + height * 0.5
                + self.cfg.object_table_clearance,
            )
            prim_path = f"{env_path}/Object"
            object_cfg = sim_utils.UsdFileCfg(
                usd_path=asset.usd_path.resolve().as_posix(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg()
                if self.cfg.override_object_physics
                else None,
                collision_props=sim_utils.CollisionPropertiesCfg()
                if self.cfg.override_object_physics
                else None,
            )
            object_cfg.func(
                prim_path,
                object_cfg,
                translation=obj_init_position,
                orientation=self.object_orientation_offset,
            )
            self.objects.append(RigidObject(RigidObjectCfg(prim_path=prim_path)))
            self.scene.rigid_objects[f"object_{env_id}"] = self.objects[-1]

        self._recording_camera = None
        if self.cfg.enable_recording_camera:
            self._recording_camera = TiledCamera(self.cfg.recording_camera)
            self.scene.sensors["recording_camera"] = self._recording_camera

    def select_grasp(
        self, env_id: int, grasp_index: int = 0
    ) -> tuple[torch.Tensor, int]:
        """Transform a cached grasp into the world frame using the current object pose.

        Args:
            env_id: Scalar environment index in ``[0, num_envs)``.
            grasp_index: Scalar grasp index, wrapped modulo the asset's grasp count.

        Returns:
            World-from-grasp homogeneous transform shaped ``(4, 4)`` and the
            resolved integer grasp index. Translation is in meters.

        Raises:
            RuntimeError: No grasps are loaded for the selected object.
        """
        asset_id = self.selected_assets[env_id].asset_id
        grasp_poses = self.object_grasps.get(asset_id)
        if grasp_poses is None:
            raise RuntimeError(f"No grasps loaded for env_{env_id} object {asset_id}.")

        pose_index = grasp_index % len(grasp_poses)
        pose = self.objects[env_id].data.root_pose_w[0]
        grasp_pose = grasp_poses[pose_index].to(pose)
        world_from_object = transform_from_pos_wxyz(pose[:3], pose[3:7])
        return world_from_object @ grasp_pose, pose_index
