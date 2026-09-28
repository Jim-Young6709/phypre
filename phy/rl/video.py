"""Periodic pushing clips using the table environment's camera and recorder."""

import subprocess
import sys
from pathlib import Path

from phy.utils.recording import DebugVideoRecorder


class PushVideoLogger:
    def __init__(self, env):
        self.env = env
        self.camera = env._recording_camera
        self.step_index = 0
        self._names = []
        cfg = env.cfg
        video_dir = (
            Path(cfg.video_dir)
            if cfg.video_dir
            else (Path(cfg.log_dir or "outputs") / "videos" / "push")
        )
        self.recorder = DebugVideoRecorder(
            True,
            video_dir,
            round(1 / (env.step_dt * cfg.video_every)),
            cfg.video_every,
            min(cfg.video_envs, env.num_envs),
        )
        self.recorder.start()

    def step(self):
        cfg = self.env.cfg
        clip_step = self.step_index % cfg.video_interval
        clip_length = min(
            cfg.video_length or self.env.max_episode_length, cfg.video_interval
        )
        if clip_step == 0:
            self._names = [
                f"step_{self.step_index:08d}_env_{i}"
                for i in range(self.recorder.max_videos)
            ]
        if clip_step < clip_length and clip_step % cfg.video_every == 0:
            self.camera.update(self.env.step_dt * cfg.video_every, force_recompute=True)
            rgb = self.camera.data.output["rgb"].detach().cpu().numpy()
            self.recorder.capture(rgb, clip_step, self._names)
        self.step_index += 1
        if clip_step == clip_length - 1:
            self._finish_clip()

    def _finish_clip(self):
        self.recorder.close()
        if not self._names:
            return
        import imageio_ffmpeg

        videos = {}
        wandb = sys.modules.get("wandb")
        for i, name in enumerate(self._names):
            path = self.recorder.video_dir / f"{name}.mp4"
            encoded = path.with_suffix(".h264.mp4")
            # The table recorder's mp4v codec needs H.264 conversion for web players.
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(encoded),
                ],
                check=True,
            )
            encoded.replace(path)
            if wandb is not None and wandb.run is not None:
                videos[f"videos/push/env_{i}"] = wandb.Video(str(path), format="mp4")
        if videos:
            wandb.run.log(
                {
                    **videos,
                    "videos/env_steps": self.step_index * self.env.num_envs,
                }
            )
        self._names = []

    def close(self):
        self._finish_clip()
