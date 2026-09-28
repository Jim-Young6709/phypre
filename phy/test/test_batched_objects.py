"""Regression checks for batched grasp access and trajectory snapshots."""

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch

from phy.franka_table_datagen import FrankaTableDatagen


@pytest.fixture
def table(monkeypatch):
    # Keep the real table logic while substituting simulator construction.
    class Base:
        def __init__(self, cfg, *args, **kwargs):
            self.scene = SimpleNamespace(env_prim_paths=cfg.env_paths)
            self.object = SimpleNamespace(
                root_physx_view=SimpleNamespace(prim_paths=cfg.body_paths)
            )

    imports = {
        "isaaclab": {},
        "isaaclab.sim": {},
        "isaaclab.assets": {"RigidObject": object, "RigidObjectCfg": object},
        "isaaclab.sensors": {"TiledCamera": object},
        "pxr": {"UsdPhysics": SimpleNamespace()},
        "phy.cfg.franka_table_env_cfg": {"FrankaTableEnvCfg": object},
        "phy.franka_base_env": {"FrankaBaseEnv": Base},
    }
    for name, attrs in imports.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "franka_table.py"
    spec = importlib.util.spec_from_file_location("batched_test_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_usd_asset_metadata", dict)
    monkeypatch.setattr(module, "discover_thor_assets", lambda _: [])
    monkeypatch.setattr(module, "select_env_assets", lambda *args: ([], {}))
    return module.FrankaTableEnv


@pytest.mark.parametrize("count", [1, 12])
def test_object_order_validation(table, count):
    paths = [f"/World/envs/env_{i}" for i in range(count)]
    bodies = [f"{path}/Object/Geometry/asset_{i}" for i, path in enumerate(paths)]
    cfg = SimpleNamespace(
        env_paths=paths, body_paths=bodies, asset_axis_convention="usd",
        scene=SimpleNamespace(num_envs=count), usd_root="unused", start_object_idx=0,
        num_grasps=0, require_grasps=False,
    )
    table(cfg)
    invalid = [bodies[:-1], bodies + [bodies[0]], bodies[:-1] + ["/World/other/Object/body"]]
    if count > 1:
        invalid.append(bodies[::-1])
    for cfg.body_paths in invalid:
        with pytest.raises(RuntimeError, match="one object rigid body per environment"):
            table(cfg)


def test_grasp_uses_selected_batch_row(table):
    env = table.__new__(table)
    env.selected_assets = [SimpleNamespace(asset_id="a"), SimpleNamespace(asset_id="b")]
    grasps = torch.eye(4).repeat(2, 1, 1)
    grasps[1, 0, 3] = 0.25
    env.object_grasps = {"b": grasps}
    env.object = SimpleNamespace(data=SimpleNamespace(root_pose_w=torch.tensor([
        [0., 0., 0., 1., 0., 0., 0.], [2., 3., 4., 1., 0., 0., 0.],
    ])))
    pose, index = env.select_grasp(1, 3)
    assert index == 1
    torch.testing.assert_close(pose[:3, 3], torch.tensor([2.25, 3., 4.]))
    with pytest.raises(RuntimeError, match="No grasps"):
        env.select_grasp(0)


@pytest.mark.parametrize("body_parents", [[], ["Geometry", "Geometry"], ["Other"]])
def test_invalid_asset_layout_fails_before_batch_creation(table, monkeypatch, body_parents):
    env = table.__new__(table)
    env_path = "/World/envs/env_0"
    env.cfg = SimpleNamespace(
        table_size=(0.8, 0.8, 0.05), table_center=(0.55, 0., 0.),
        scene=SimpleNamespace(num_envs=1), default_object_height=0.2,
        object_xy_offset=(0., 0.), object_table_clearance=0.005,
        override_object_physics=False,
    )
    env.scene = SimpleNamespace(env_prim_paths=[env_path])
    env.selected_assets = [SimpleNamespace(asset_id="bad_asset", usd_path=Path("bad.usd"))]
    env.asset_metadata, env.object_orientation_offset = {}, torch.tensor([1., 0., 0., 0.])
    bodies = []
    for parent in body_parents:
        body = MagicMock()
        body.GetParent.return_value.GetPath.return_value = f"{env_path}/Object/{parent}"
        bodies.append(body)
    sim = MagicMock()
    sim.get_all_matching_child_prims.return_value = bodies
    monkeypatch.setitem(table._setup_task_scene.__globals__, "sim_utils", sim)
    with pytest.raises(RuntimeError, match="bad_asset.*exactly one rigid body"):
        env._setup_task_scene()


@pytest.mark.parametrize("count", [1, 3])
def test_lift_detection_keeps_initial_snapshot(count, tmp_path):
    gen = FrankaTableDatagen.__new__(FrankaTableDatagen)
    pose = torch.zeros(count, 7)
    pose[:, 2] = 0.2
    gen.env = SimpleNamespace(
        object=SimpleNamespace(data=SimpleNamespace(root_pose_w=pose)), num_envs=count,
        sim=SimpleNamespace(set_camera_view=lambda **kwargs: None),
        reset=lambda: None, close=lambda: None,
    )
    gen.select_grasp_poses = gen.initialize_pregrasp = lambda: None
    gen.collect_trajectories = lambda: pose[0, 2].add_(0.1)
    gen.max_demos, gen.first_demo_id = count, 7
    gen.h5_file = SimpleNamespace(flush=lambda: None)
    gen._grasp_debug_draw = None
    gen.recorder = SimpleNamespace(close=lambda: None)
    saved, reports = [], []
    gen.write_demo = lambda env_id, demo_id: saved.append((env_id, demo_id))
    gen.write_batch_report = lambda path, gains, successful, saved: reports.append(
        (gains.clone(), successful, saved)
    )
    assert gen.run(tmp_path / "report.txt") == 1
    assert saved == [(0, 7)]
    expected = torch.zeros(count)
    expected[0] = 0.1
    torch.testing.assert_close(reports[0][0], expected)
    assert reports[0][1:] == ([0], [0])


@pytest.mark.parametrize("count", [1, 3])
def test_recorded_poses_are_snapshots_and_keep_environment_axis(count, tmp_path):
    gen = FrankaTableDatagen.__new__(FrankaTableDatagen)
    pose = torch.zeros(count, 7)
    pose[:, 0] = torch.arange(count)
    pose[:, 3] = 1
    gen.env = SimpleNamespace(
        object=SimpleNamespace(data=SimpleNamespace(root_pose_w=pose)),
        robot_dof_targets=torch.zeros(count, 8), logical_dof_indices=list(range(8)),
        gripper_dof_index=7, _eef_body_id=0,
        robot=SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.zeros(count, 8), joint_vel=torch.zeros(count, 8),
            body_pose_w=pose[:, None].clone(),
        )),
        selected_assets=[SimpleNamespace(asset_id=f"asset_{i}", usd_path=Path(f"{i}.usd"))
                         for i in range(count)],
        scene=SimpleNamespace(env_prim_paths=[f"/World/envs/env_{i}" for i in range(count)]),
        cfg=SimpleNamespace(asset_axis_convention="usd"),
    )
    gen.buffer = defaultdict(list)
    gen.grasp_pose_w = gen.pregrasp_pose_w = pose.clone()
    gen.grasp_indices = [0] * count
    before = pose.numpy().copy()
    gen.record_step(pose.clone(), 0)
    pose[:, 2] += 0.1
    gen.record_step(pose.clone(), 2)
    np.testing.assert_array_equal(gen.buffer["obs/object_pose"][0], before)
    with h5py.File(tmp_path / "demos.h5", "w") as gen.h5_file:
        for i in range(count):
            gen.write_demo(i, i)
            demo = gen.h5_file[f"data/demo_{i}"]
            assert demo["obs/object_pose"].shape == (2, 7)
            np.testing.assert_array_equal(demo["obs/object_pose"][0], before[i])
            np.testing.assert_array_equal(demo["obs/object_pose"][1], pose[i].numpy())
            np.testing.assert_array_equal(demo["object_pose"][:], pose[i].numpy())
            assert demo.attrs["asset_id"] == f"asset_{i}"
