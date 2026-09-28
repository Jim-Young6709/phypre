Train asymmetric PPO with the existing Isaac Lab rl_games runner. The actor sees
object/goal poses, the object's axis-aligned size, and the previous action, all
expressed in the current end-effector frame. It has no joint observations. A
separate privileged critic also receives arm joint positions/velocities,
robot-relative end-effector pose, object velocities, and remaining episode time.
Actions are six end-effector-local motion commands; the gripper stays closed.
Targets vary in table XY and yaw.

The default feedforward version uses `phy/cfg/franka_push_ppo.yaml`. Use `--lstm`
for `phy/cfg/franka_push_ppo_lstm.yaml`, which uses DEXTRAH's one-layer LSTMs:
actor 1024 units before a [512, 512] MLP, privileged critic 2048 units after a
[1024, 512] MLP, with layer normalization and input/output concatenation.
Sequences contain 16 steps and hidden states reset when episodes end. Actor and
critic update minibatches are capped at 1024 transitions (16 per environment
for small runs), so their training activations do not grow with the environment
count. Rollout buffers and recurrent states still grow with that count.

```bash
python phy/rl/run.py train --headless --lstm --wandb
python phy/rl/run.py play --lstm --checkpoint /path/to/lstm_checkpoint.pth
```

Omit `--lstm` (or use `--no-lstm`) for feedforward training/playback. Each YAML
has its own W&B settings; the LSTM version uses a `franka_push_lstm_` run prefix
and writes local logs to `logs/rl_games/franka_push_lstm`.

Objects are filtered to fit the pushing workspace, then settled once at startup
to obtain a supported reference pose. Episode resets randomize XY/yaw from that
pose without stepping other environments. The task terminates on settled success,
leaving the table, or timeout. Reaching uses TCP-to-object-center distance, with
the TCP offset matching the data-generation setup. There are no success bonuses
or failure penalties.

Use your existing Isaac Lab conda environment. From the phypre directory, install
phypre without changing its already-installed dependencies, then install the
rl_games fork specified by the installed Isaac Lab package:

```bash
conda activate phy_isaaclab
python -m pip install --no-deps -e .
python -m pip install 'rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11'
```

From the phypre directory (1024 environments by default):

```bash
python phy/rl/run.py train --headless --num_envs 64
python phy/rl/run.py train --headless --num_envs 4 --max_iterations 2
python phy/rl/run.py play --num_envs 4 --checkpoint /path/to/checkpoint.pth
```

For W&B, set `wandb.entity` (your username/team) and optionally `wandb.project`
and `wandb.name` in `phy/cfg/franka_push_ppo.yaml`. Run `wandb login` once, then:

```bash
python phy/rl/run.py train --headless --wandb
python phy/rl/run.py train --headless --no-wandb
```

W&B is off when the flag is omitted. These W&B YAML settings are read by the
launcher before forwarding arguments to Isaac Lab. The run name supports
`strftime` formatting; `franka_push_%Y-%m-%d_%H-%M-%S` includes the local launch
date and time.

`--wandb` also enables video logging using the existing Franka table camera and
recorder. The first four environments have recording cameras, positioned one meter along each
positive XYZ axis from the table surface center and aimed at that center. Clips
use 640×480 images and Isaac Lab's performance rendering preset. Override with
`env.recording_camera.width=1280 env.recording_camera.height=960 --rendering_mode balanced`
for higher-quality rendering. Clips
are converted to H.264 for browser playback and uploaded to `videos/push/env_0` through `env_3` in the
same W&B run. Local copies are saved under the run's `videos/push/` directory.
The default clip covers the full episode duration (10 seconds / 600 policy steps)
every 12,000 policy steps, sampled every two steps (30 fps). `--video_envs`
controls the camera count; `--video_length 0` derives clip length from the episode
configuration. These are policy steps, with two physics substeps per policy step. The last partial clip is saved when training closes.

Training randomizes the initial episode progress buffer to stagger timeouts.
Later episode resets start at zero. Playback retains zero initial progress;
`env.randomize_initial_progress=false` disables training randomization.

```bash
# W&B metrics and video
python phy/rl/run.py train --headless --wandb --num_envs 64
# Local video without W&B
python phy/rl/run.py train --headless --video --num_envs 64
# W&B metrics without video
python phy/rl/run.py train --headless --wandb --no-video
# Short test with completed episodes, a video, and W&B uploads
python phy/rl/run.py train --headless --wandb --num_envs 4 --max_iterations 2 --video_length 60 --video_interval 60 env.episode_length_s=0.5
```

W&B receives `Episode/success_rate`, `Episode/failure_rate`, and
`Episode/Reward/total`, plus the separate weighted `reach`, `position`,
`orientation`, and `action_rate` episode returns under `Episode/Reward/`.
Success rates are averaged over completed episodes, including timeouts; initial
resets are excluded. The reward components sum to the total return. Native PPO
reward/loss metrics continue to sync through TensorBoard. The video and metric
options also apply to `--lstm`.

Isaac Lab libraries come from the active Python environment. The launcher only
reads train/play scripts from the sibling `IsaacLab` checkout; the conda package
does not include those scripts. It does not add that checkout's libraries to
Python's import path. `ISAACLAB_PATH` overrides the location of these scripts.
All remaining arguments and Hydra overrides pass through
to Isaac Lab, for example `env.episode_length_s=10` or
`agent.params.config.learning_rate=0.0001`. Feedforward PPO and critic minibatches
scale with the environment count; LSTM minibatches use the cap described above.
Logs and checkpoints are in `logs/rl_games/franka_push`.

Profile GPU and host memory by initialization/training phase with:

```bash
python phy/rl/profile_memory.py --report logs/push_memory.jsonl --headless --num_envs 1024 --no-video --max_iterations 3
```

Use `--video` and/or `--lstm` for comparisons. Profiling disables W&B.
The JSONL report separates total process GPU memory from PyTorch's live tensors
and reserved cache, and samples peak process GPU usage. All phypre USD loaders
check local paths directly and await remote checks without Isaac Lab's fixed
0.1-second polling delay. Remote timeouts remain in place; the installed Isaac
Lab files remain unchanged.

RTX resources stay allocated between video clips. The pushing configuration
reserves 2**20 entries for PhysX's found/lost aggregate-pair buffer instead of
the installed default 2**25, saving approximately 1 GiB of VRAM per process.

Training requires Isaac Sim, an NVIDIA GPU, and the configured Franka/table/object
assets. The two-iteration command checks training and checkpoint creation; it is
not enough to learn pushing.
