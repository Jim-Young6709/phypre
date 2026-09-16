"""Check all rigid objects in one batch, with one object and table per environment.

Run from phypre in the Isaac Lab Python environment:
    python -m phy.test.test_rigid_object_stability --headless

Output: one ``asset_id True/False`` line per object. True means its center of
mass moved more than 0.05 m from its spawn position during 500 physics steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("rigid_object_stability.txt"))
    parser.add_argument("--steps", type=int, default=500)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    simulation_app = AppLauncher(args).app

    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import build_simulation_context
        from pxr import Gf, Usd, UsdGeom

        from phy.cfg.franka_table_env_cfg import FrankaTableEnvCfg
        from phy.utils.assets import discover_thor_assets
        from phy.utils.transforms import axis_rotation_wxyz

        cfg = FrankaTableEnvCfg()
        cfg.sim.device = args.device
        assets = discover_thor_assets(args.usd_root or cfg.usd_root)
        if not assets:
            raise RuntimeError(f"No rigid objects found in {args.usd_root or cfg.usd_root}")
        orientation = tuple(axis_rotation_wxyz(cfg.asset_axis_convention).tolist())
        table_top = cfg.table_center[2] + cfg.table_size[2] / 2
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cfg.scene.num_envs = len(assets)
        with build_simulation_context(sim_cfg=cfg.sim, auto_add_lighting=True) as sim:
            # Allow cleanup without waiting for GUI playback to resume.
            if sim._app_control_on_stop_handle is not None:
                sim._app_control_on_stop_handle.unsubscribe()
                sim._app_control_on_stop_handle = None
            scene = InteractiveScene(cfg.scene)
            table_cfg = sim_utils.CuboidCfg(
                size=cfg.table_size,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.6
                ),
            )
            for env_path, asset in zip(scene.env_prim_paths, assets, strict=True):
                table_cfg.func(f"{env_path}/Table", table_cfg, translation=cfg.table_center)
                object_cfg = sim_utils.UsdFileCfg(usd_path=str(asset.usd_path))
                prim = object_cfg.func(
                    f"{env_path}/Object", object_cfg,
                    translation=(0.0, 0.0, 0.0), orientation=orientation,
                )
                # Measure in the environment frame, including THOR's guide colliders.
                bounds = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    ["default", "render", "proxy", "guide"],
                    False,  # Compute bounds from geometry, not extents hints.
                    True,  # Include invisible collision geometry.
                ).ComputeRelativeBound(
                    prim, sim.stage.GetPrimAtPath(env_path)
                ).ComputeAlignedRange()
                if bounds.IsEmpty():
                    raise RuntimeError(f"Empty object bounds: {asset.asset_id}")
                center = bounds.GetMidpoint()
                prim.GetAttribute("xformOp:translate").Set(
                    Gf.Vec3d(
                        cfg.table_center[0] - center[0],
                        cfg.table_center[1] - center[1],
                        table_top - bounds.GetMin()[2],
                    )
                )
            scene.filter_collisions()
            objects = RigidObject(
                RigidObjectCfg(prim_path="/World/envs/env_.*/Object/Geometry/.*")
            )
            sim.reset()
            # PhysX view ordering need not match the asset list's ordering.
            env_ids = [
                int(path.split("/")[3].removeprefix("env_"))
                for path in objects.root_physx_view.prim_paths
            ]
            if sorted(env_ids) != list(range(len(assets))):
                raise RuntimeError("Expected exactly one rigid body per object environment")
            spawn_centers = objects.data.root_com_pos_w.clone()
            if not torch.isfinite(spawn_centers).all():
                raise RuntimeError("Invalid spawn centers")
            moved = torch.zeros(len(assets), dtype=torch.bool, device=sim.device)
            print(f"Testing {len(assets)} objects in {scene.num_envs} environments", flush=True)
            for step in range(args.steps):
                if not simulation_app.is_running():
                    raise RuntimeError("Simulation closed before all steps completed")
                sim.step(render=not args.headless)
                objects.update(sim.get_physics_dt())
                positions = objects.data.root_com_pos_w
                if not torch.isfinite(positions).all():
                    raise RuntimeError(f"Non-finite object positions at step {step + 1}")
                moved |= torch.linalg.vector_norm(positions - spawn_centers, dim=-1) > 0.05
                if (step + 1) % 100 == 0:
                    print(f"Step {step + 1}/{args.steps}: {moved.sum().item()} moved >5 cm", flush=True)
            labels = dict(zip(env_ids, moved.tolist(), strict=True))
            with output_path.open("w", encoding="utf-8") as output:
                for env_id, asset in enumerate(assets):
                    output.write(f"{asset.asset_id} {labels[env_id]}\n")
        print(f"Saved {len(assets)} object labels to {output_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
