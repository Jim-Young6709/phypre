"""Profile the existing RL launcher without changing the task or installed libraries.

Example: python phy/rl/profile_memory.py --report logs/memory.jsonl --headless
         --num_envs 64 --no-video --max_iterations 3
All arguments after --report are forwarded to phy/rl/run.py train.
"""

import argparse
import functools
import gc
import json
import os
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

from phy.rl import run


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--report", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("")
    process = psutil.Process()
    started = time.monotonic()
    peak_process_mib = 0.0
    stopped = threading.Event()

    def process_gpu_mib():
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return sum(
            float(line.split(",")[1])
            for line in output.splitlines()
            if int(line.split(",")[0]) == os.getpid()
        )

    def monitor():
        nonlocal peak_process_mib
        while not stopped.is_set():
            peak_process_mib = max(peak_process_mib, process_gpu_mib())
            stopped.wait(0.5)

    def snapshot(stage, **details):
        nonlocal peak_process_mib
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 2**20
            reserved = torch.cuda.memory_reserved() / 2**20
            peak_allocated = torch.cuda.max_memory_allocated() / 2**20
        else:
            allocated = reserved = peak_allocated = 0.0
        gpu = process_gpu_mib()
        peak_process_mib = max(peak_process_mib, gpu)
        row = {
            "stage": stage,
            "elapsed_s": round(time.monotonic() - started, 3),
            "process_gpu_mib": gpu,
            "peak_process_gpu_mib": peak_process_mib,
            "torch_allocated_mib": round(allocated, 3),
            "torch_reserved_mib": round(reserved, 3),
            "torch_peak_allocated_mib": round(peak_allocated, 3),
            "non_torch_reserved_mib": round(gpu - reserved, 3),
            "rss_mib": round(process.memory_info().rss / 2**20, 3),
            **details,
        }
        with args.report.open("a") as output:
            output.write(json.dumps(row) + "\n")
        print("MEMORY " + json.dumps(row), flush=True)

    def wrap(owner, name, stage, *, once=True, reset_peak=False):
        original = getattr(owner, name)
        calls = 0

        @functools.wraps(original)
        def measured(*a, **kw):
            nonlocal calls
            calls += 1
            if once and calls > 1:
                return original(*a, **kw)
            label = stage if once else f"{stage}_{calls}"
            if reset_peak:
                torch.cuda.reset_peak_memory_stats()
            snapshot(label + "/before")
            result = original(*a, **kw)
            snapshot(label + "/after")
            return result

        setattr(owner, name, measured)

    def install_hooks():
        from isaaclab.sensors import TiledCamera
        from isaaclab.sim import SimulationContext
        from rl_games.algos_torch.a2c_continuous import A2CAgent
        from rl_games.algos_torch.central_value import CentralValueTrain

        import phy.franka_base_env as base
        from phy.franka_push import FrankaPushEnv
        from phy.franka_table import FrankaTableEnv
        from phy.rl.video import PushVideoLogger

        snapshot("task_imported")
        wrap(base.FrankaBaseEnv, "_setup_scene", "scene")
        wrap(FrankaTableEnv, "_setup_task_scene", "objects_and_camera_prims")
        wrap(SimulationContext, "reset", "physics_and_sensor_initialization")
        wrap(TiledCamera, "_initialize_impl", "camera_initialization")
        wrap(base.IKSolverConfig, "load_from_robot_config", "curobo_config")
        wrap(base.IKSolver, "__init__", "curobo_solver")
        wrap(FrankaPushEnv, "_settle_objects", "object_settling")
        wrap(FrankaPushEnv, "__init__", "environment")
        wrap(A2CAgent, "__init__", "ppo_models")
        wrap(A2CAgent, "init_tensors", "ppo_rollout_buffers")
        wrap(A2CAgent, "train_epoch", "ppo_epoch", once=False, reset_peak=True)
        wrap(A2CAgent, "play_steps", "rollout")
        wrap(A2CAgent, "play_steps_rnn", "recurrent_rollout")
        wrap(CentralValueTrain, "train_net", "critic_update")
        wrap(A2CAgent, "train_actor_critic", "actor_update")
        wrap(PushVideoLogger, "step", "first_video_frame")
        wrap(FrankaPushEnv, "close", "environment_close")

    from isaaclab.app import AppLauncher

    wrap(AppLauncher, "__init__", "simulation_app")
    original_run_path = runpy.run_path

    def profiled_run_path(path, *a, **kw):
        scope = original_run_path(path, *a, **kw)
        snapshot("runner_imported")
        install_hooks()
        original_main = scope["main"]

        @functools.wraps(original_main)
        def profiled_main(*main_args, **main_kwargs):
            try:
                return original_main(*main_args, **main_kwargs)
            finally:
                snapshot("training_finished")
                gc.collect()
                torch.cuda.empty_cache()
                snapshot("after_gc_and_empty_cache")
                stopped.set()

        scope["main"] = profiled_main
        return scope

    runpy.run_path = profiled_run_path
    sys.argv = [str(Path(run.__file__)), "train", "--no-wandb", *remaining]
    snapshot("before_launch", command=sys.argv)
    threading.Thread(target=monitor, daemon=True).start()
    run.main()


if __name__ == "__main__":
    main()
