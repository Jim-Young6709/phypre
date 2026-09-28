"""Register phypre tasks and run Isaac Lab's existing rl_games train/play script."""

import argparse
import inspect
import os
import runpy
import sys
from pathlib import Path
from time import strftime

import yaml

import phy  # noqa: F401 -- register tasks before Isaac Lab resolves their configuration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("mode", choices=("train", "play"))
    parser.add_argument("--lstm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--video_length", type=int, default=0)
    parser.add_argument("--video_interval", type=int, default=12000)
    parser.add_argument("--video_envs", type=int, default=4)
    args, remaining = parser.parse_known_args()
    config_name = "franka_push_ppo_lstm.yaml" if args.lstm else "franka_push_ppo.yaml"
    agent_key = (
        "rl_games_lstm_cfg_entry_point" if args.lstm else "rl_games_cfg_entry_point"
    )
    wandb_args = []
    if args.wandb:
        config_path = Path(__file__).resolve().parents[1] / "cfg" / config_name
        config = yaml.safe_load(config_path.read_text())["wandb"]
        if not config["entity"]:
            parser.error(f"Set wandb.entity in phy/cfg/{config_name} first.")
        if config["name"] is not None:
            config["name"] = strftime(config["name"])
        wandb_args = ["--track"]
        for key, flag in (
            ("project", "--wandb-project-name"),
            ("entity", "--wandb-entity"),
            ("name", "--wandb-name"),
        ):
            if config[key] is not None:
                wandb_args.extend((flag, str(config[key])))
    video_args = []
    record_video = args.wandb if args.video is None else args.video
    if record_video:
        # Use the table task's camera/recorder instead of the runner's viewport recorder.
        video_args = [
            "--enable_cameras",
            "--rendering_mode",
            "performance",
            "env.enable_recording_camera=true",
            f"env.video_length={args.video_length}",
            f"env.video_interval={args.video_interval}",
            f"env.video_envs={args.video_envs}",
        ]
    train_args = ["env.randomize_initial_progress=true"] if args.mode == "train" else []
    isaaclab_path = Path(
        os.environ.get(
            "ISAACLAB_PATH", Path(__file__).resolve().parents[3] / "IsaacLab"
        )
    )
    script = (
        isaaclab_path
        / "scripts"
        / "reinforcement_learning"
        / "rl_games"
        / f"{args.mode}.py"
    )
    sys.argv = [
        str(script),
        "--task",
        "Phy-Franka-Push-Direct-v0",
        "--agent",
        agent_key,
        *wandb_args,
        *video_args,
        *train_args,
        *remaining,
    ]
    runner = runpy.run_path(str(script))
    if args.mode == "train":

        class FlushingRunner(runner["Runner"]):
            def load(self, config):
                ppo = config["params"]["config"]
                if args.lstm:
                    for settings in (ppo, ppo["central_value_config"]):
                        settings["minibatch_size"] = min(
                            settings["minibatch_size"],
                            ppo["num_actors"] * settings["minibatch_size_per_env"],
                        )
                super().load(config)

            def run_train(self, args):
                super().run_train(args)
                self.algo_observer.writer.close()

        # Flush TensorBoard's final metrics before finishing the W&B sync.
        inspect.unwrap(runner["main"]).__globals__["Runner"] = FlushingRunner
    try:
        runner["main"]()
    finally:
        # Isaac Sim's fast shutdown bypasses Python's W&B exit hook.
        wandb = sys.modules.get("wandb")
        if wandb is not None and wandb.run is not None:
            wandb.finish()
        runner["simulation_app"].close()


if __name__ == "__main__":
    main()
