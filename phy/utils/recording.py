"""
Reusable helpers for trajectory and video recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


class DebugVideoRecorder:
    def __init__(
        self, enabled: bool, video_dir: Path, fps: int, every: int, max_videos: int
    ):
        self.enabled = enabled
        self.video_dir = video_dir
        self.fps = fps
        self.every = max(1, every)
        self.max_videos = max(0, max_videos)
        self._cv2: Any = None
        self._writers: dict[str, Any] = {}

    def start(self) -> None:
        if not self.enabled:
            return
        import cv2

        self._cv2 = cv2
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        rgb: np.ndarray,
        step_index: int,
        video_names: list[str],
    ) -> None:
        if not self.enabled or self._cv2 is None or step_index % self.every:
            return
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8, copy=False)

        for index, video_name in enumerate(video_names[: self.max_videos]):
            frame = rgb[index, ..., :3]
            if video_name not in self._writers:
                height, width = frame.shape[:2]
                path = self.video_dir / f"{video_name}.mp4"
                codec = self._cv2.VideoWriter_fourcc(*"mp4v")
                self._writers[video_name] = self._cv2.VideoWriter(
                    path.as_posix(), codec, self.fps, (width, height)
                )
            self._writers[video_name].write(
                self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
            )

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()


def next_demo_index(data_group: h5py.Group) -> int:
    indices = [
        int(key.removeprefix("demo_"))
        for key in data_group
        if key.startswith("demo_") and key.removeprefix("demo_").isdigit()
    ]
    return max(indices, default=-1) + 1
