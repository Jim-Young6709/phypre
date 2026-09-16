"""CPU checks of pushing task tensors; simulator construction is excluded."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def push():
    """Use real Isaac Lab math with only simulator-facing imports substituted."""
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    class Table:
        def _reset_idx(self, env_ids):
            self.actions[env_ids] = 0
            self.episode_length_buf[env_ids] = 0

        def _pre_physics_step(self, actions):
            self.actions = actions.clamp(-1, 1)

    root = Path(__file__).resolve().parents[1]
    lab = Path(importlib.util.find_spec("isaaclab").origin).parent
    math_path = next(lab.glob("**/isaaclab/utils/math.py"), lab / "utils/math.py")
    math_module = load("push_test_math", math_path)
    with pytest.MonkeyPatch.context() as patch:
        imports = {
            "isaaclab.utils": {"configclass": lambda cls: cls},
            "isaaclab.scene": {"InteractiveSceneCfg": SimpleNamespace},
            "phy.franka_table": {"FrankaTableEnv": Table},
            "phy.cfg.franka_table_env_cfg": {"FrankaTableEnvCfg": type("Cfg", (), {
                "table_center": (0.55, 0.0, 0.0), "table_size": (0.8, 0.8, 0.05),
                "dof_velocity_scale": 0.1,
                "use_robotiq_gripper": True,
            })},
        }
        for name, attributes in imports.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            patch.setitem(sys.modules, name, module)
        patch.setitem(sys.modules, "isaaclab.utils.math", math_module)
        cfg = load("phy.cfg.franka_push_env_cfg", root / "cfg/franka_push_env_cfg.py")
        patch.setitem(sys.modules, "phy.cfg.franka_push_env_cfg", cfg)
        yield load("push_test_task", root / "franka_push.py")


@pytest.fixture
def env(push):
    env = push.FrankaPushEnv.__new__(push.FrankaPushEnv)
    env.cfg = push.FrankaPushEnvCfg()
    env.device, env.num_envs, env._eef_body_id = "cpu", 2, 0
    env.max_episode_length = 480
    env.arm_dof_indices = list(range(7))
    env.scene = SimpleNamespace(env_origins=torch.tensor([[0., 0., 0.], [2., 3., 1.]]))
    eef = torch.tensor([[0.4, 0.1, 0.3, 1., 0., 0., 0.]]).repeat(2, 1)
    eef[:, :3] += env.scene.env_origins
    env.robot = SimpleNamespace(_ALL_INDICES=torch.arange(2), data=SimpleNamespace(
        body_pose_w=eef[:, None], body_quat_w=eef[:, None, 3:],
        root_pose_w=torch.cat((env.scene.env_origins, eef[:, 3:]), dim=-1),
        joint_pos=torch.zeros(2, 7), joint_vel=torch.zeros(2, 7),
    ))
    pose = torch.tensor([[0.55, 0., 0.10, 1., 0., 0., 0.]]).repeat(2, 1)
    pose[:, :3] += env.scene.env_origins
    velocity = torch.zeros(2, 6)
    env.object = SimpleNamespace(data=SimpleNamespace(
        root_pose_w=pose, root_vel_w=velocity,
        root_lin_vel_w=velocity[:, :3], root_ang_vel_w=velocity[:, 3:],
    ))
    env.object.write_root_pose_to_sim = lambda value, env_ids: pose.__setitem__(env_ids, value)
    env.object.write_root_velocity_to_sim = lambda value, env_ids: velocity.__setitem__(env_ids, value)
    env.robot_dof_lower_limits, env.robot_dof_upper_limits = -torch.ones(2, 7), torch.ones(2, 7)
    env.object_size = torch.tensor([[0.08, 0.12, 0.14], [0.12, 0.08, 0.14]])
    env.object_center = torch.tensor([[0.02, 0., 0.], [-0.03, 0.01, 0.]])
    env.goal_pose = pose.clone()
    env.support_quat, env.support_height = pose[:, 3:].clone(), torch.full((2,), 0.10)
    env.action_frame_quat = eef[:, 3:].clone()
    env.actions = torch.zeros(2, 6)
    env.action_rate = torch.zeros(2)
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.success_steps = torch.zeros(2, dtype=torch.long)
    env.success = torch.zeros(2, dtype=torch.bool)
    env.failure = torch.zeros(2, dtype=torch.bool)
    env.episode_reward_sums = {
        name: torch.zeros(2)
        for name in ("reach", "position", "orientation", "action_rate")
    }
    env.table_top, env.extras = 0.025, {}
    return env


def test_randomized_initial_progress_updates_critic(env, push, monkeypatch):
    monkeypatch.setattr(push.FrankaTableEnv, "reset",
        lambda self, **kwargs: (self._get_observations(), self.extras), raising=False)
    env.cfg.randomize_initial_progress = True
    with torch.random.fork_rng():
        torch.manual_seed(42)
        observations, extras = env.reset()
    assert (env.episode_length_buf > 0).all()
    assert (env.episode_length_buf < env.max_episode_length).all()
    torch.testing.assert_close(observations["critic"][:, -1],
        1 - env.episode_length_buf / env.max_episode_length)
    assert extras == {}
    env._reset_idx(torch.tensor([0]))
    assert env.episode_length_buf[0] == 0  # later episodes retain their full duration


def test_actor_frame_invariance_and_privileged_isolation(env, push):
    env.actions[:] = torch.tensor([0.3, -0.2, 0.1, -0.1, 0.4, 0.2])
    before = env._get_observations()
    assert before["policy"].shape == (2, 23)
    assert before["critic"].shape == (2, 51)
    angles = torch.tensor([0.3, -0.5, 0.8])
    rotation = push.quat_from_euler_xyz(*angles).repeat(2, 1)
    translation = torch.tensor([4., -3., 2.])
    for pose in (env.robot.data.body_pose_w[:, 0], env.robot.data.root_pose_w,
                 env.object.data.root_pose_w, env.goal_pose):
        pose[:, :3] = push.quat_apply(rotation, pose[:, :3]) + translation
        pose[:, 3:] = push.quat_mul(rotation, pose[:, 3:])
    env.action_frame_quat = push.quat_mul(rotation, env.action_frame_quat)
    torch.testing.assert_close(env._get_observations()["policy"], before["policy"], atol=1e-6, rtol=1e-5)
    env.robot.data.joint_pos += 0.2
    env.robot.data.joint_vel += 2
    env.object.data.root_vel_w += 0.5
    after = env._get_observations()
    torch.testing.assert_close(after["policy"], before["policy"], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(after["critic"], before["critic"])


def test_previous_actions_follow_current_eef_frame(env, push):
    env.actions[:, 0] = 1
    env.actions[:, 4] = 1
    rotation = push.quat_from_euler_xyz(torch.tensor(0.), torch.tensor(0.), torch.tensor(torch.pi / 2)).repeat(2, 1)
    local = env._actions_in_frame(rotation)
    torch.testing.assert_close(local, torch.tensor([[0., -1., 0., 1., 0., 0.]]).repeat(2, 1), atol=1e-6, rtol=0)
    env.robot.data.body_quat_w[:] = rotation[:, None]
    env._pre_physics_step(local)
    torch.testing.assert_close(env.action_rate, torch.zeros(2), atol=1e-10, rtol=0)
    assert env.actions.shape == (2, 7)
    assert (env.actions[:, -1] == 1).all()
    assert env._get_observations()["policy"].shape == (2, 23)
    env.cfg.use_robotiq_gripper = False
    env._pre_physics_step(local)
    assert (env.actions[:, -1] == -1).all()


def test_assets_fit_after_tipping(env, push, monkeypatch):
    assets = [SimpleNamespace(asset_id=name) for name in ("block", "bat")]
    monkeypatch.setattr(push.FrankaTableEnv, "_get_assets", lambda self, cfg: assets, raising=False)
    env.cfg.default_object_height = 0.2
    env.asset_metadata = {
        "block": {"bbox_size": [0.08, 0.08, 0.08]},
        "bat": {"bbox_size": [0.07, 0.95, 0.07]},
    }
    assert env._get_assets(env.cfg) == assets[:1]


def test_success_requires_settling_and_failure_terminates(env):
    env.goal_pose[:, 3:] *= -1  # q and -q are the same target rotation.
    env.object.data.root_lin_vel_w[0, 0] = 1
    for _ in range(env.cfg.success_hold_steps - 1):
        assert not env._get_dones()[0].any()
    terminated, _ = env._get_dones()
    assert terminated.tolist() == [False, True]
    env.object.data.root_vel_w.zero_()
    for _ in range(env.cfg.success_hold_steps):
        terminated, _ = env._get_dones()
    assert terminated.all()
    solved_reward = env._get_rewards().clone()
    env.goal_pose[0, 0] += 0.15
    env.goal_pose[1, 3:] = torch.tensor([0., 0., 0., 1.])
    assert not env._get_dones()[0].any()
    assert (env._get_rewards() < solved_reward).all()
    env.object.data.root_pose_w[0, 0] = 1.1
    env.object.data.root_pose_w[1, 2] = env.scene.env_origins[1, 2] - 0.1
    assert env._get_dones()[0].all()
    assert env.failure.all()
    env.episode_length_buf[:] = env.max_episode_length - 1
    assert env._get_dones()[1].all()


def test_partial_reset_and_sampled_bounds(env, push):
    env.actions.fill_(0.5)
    env.object.data.root_vel_w.fill_(1)
    env.success_steps.fill_(5)
    env.action_rate.fill_(2)
    env.episode_length_buf.fill_(30)
    watched = [env.object.data.root_pose_w, env.object.data.root_vel_w, env.goal_pose,
               env.actions, env.action_rate, env.success_steps, env.episode_length_buf]
    untouched = [tensor[0].clone() for tensor in watched]
    env._reset_idx([1])
    for tensor, previous in zip(watched, untouched):
        torch.testing.assert_close(tensor[0], previous)
    assert not env.object.data.root_vel_w[1].any()
    assert not env.actions[1].any()
    assert env.success_steps[1] == env.episode_length_buf[1] == env.action_rate[1] == 0
    for _ in range(10):
        pose = env._sample_pose(torch.arange(2))
        center = pose[:, :3] + push.quat_apply(pose[:, 3:], env.object_center) - env.scene.env_origins
        half = (push.matrix_from_quat(pose[:, 3:]).abs() @ (env.object_size / 2).unsqueeze(-1)).squeeze(-1)
        lower = torch.tensor([env.cfg.workspace_x[0], env.cfg.workspace_y[0]]) + env.cfg.table_margin
        upper = torch.tensor([env.cfg.workspace_x[1], env.cfg.workspace_y[1]]) - env.cfg.table_margin
        assert ((center[:, :2] - half[:, :2]) >= lower - 1e-6).all()
        assert ((center[:, :2] + half[:, :2]) <= upper + 1e-6).all()
        torch.testing.assert_close(pose[:, 2] - env.scene.env_origins[:, 2], env.support_height)


def test_episode_rewards_accumulate_and_reset_independently(env):
    env._reset_idx(None)
    assert "log" not in env.extras  # Construction/reset is not a completed episode.
    env._get_dones()
    env.action_rate[:] = torch.tensor([1., 2.])
    reward = env._get_rewards().clone()
    env._get_rewards()
    torch.testing.assert_close(sum(env.episode_reward_sums.values()), 2 * reward)
    assert (env.episode_reward_sums["action_rate"] < 0).all()
    remaining = {name: values[0].clone() for name, values in env.episode_reward_sums.items()}
    env.episode_length_buf[:] = 2
    env.success[1] = True
    env._reset_idx([1])
    metrics = env.extras["log"]
    assert metrics["success_rate"].tolist() == [1.]
    torch.testing.assert_close(metrics["Reward/total"], 2 * reward[1:])
    torch.testing.assert_close(
        metrics["Reward/total"],
        sum(metrics[f"Reward/{name}"] for name in env.episode_reward_sums),
    )
    for name, values in env.episode_reward_sums.items():
        torch.testing.assert_close(values[0], remaining[name])
        assert values[1] == 0
    env._get_dones()
    env._get_rewards()
    assert "log" not in env.extras  # No duplicate episode metrics on subsequent steps.


def test_rl_games_observer_weights_finished_episodes(env):
    observer_module = pytest.importorskip("rl_games.common.algo_observer")
    scalars = {}
    observer = observer_module.IsaacAlgoObserver()
    observer.after_init(SimpleNamespace(
        games_to_track=100, ppo_device="cpu", device="cpu",
        writer=SimpleNamespace(add_scalar=lambda name, value, step: scalars.__setitem__(name, float(value))),
    ))
    env.episode_length_buf[:] = 1
    env.success[0] = True
    env._reset_idx([0])
    observer.process_infos({"episode": env.extras["log"]}, torch.tensor([0]))
    env.episode_length_buf[:] = 1
    env.failure[1] = True
    env._reset_idx([0, 1])
    observer.process_infos({"episode": env.extras["log"]}, torch.tensor([0, 1]))
    observer.after_print_stats(3, 1, 0.1)
    assert scalars["Episode/success_rate"] == pytest.approx(1 / 3)
    assert scalars["Episode/failure_rate"] == pytest.approx(1 / 3)
    assert {f"Episode/Reward/{name}" for name in env.episode_reward_sums} <= scalars.keys()
