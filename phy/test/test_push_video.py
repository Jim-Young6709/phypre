"""Exercise real MP4 encoding and clip upload without starting Isaac Sim."""

import sys
from types import SimpleNamespace

import cv2
import torch

from phy.rl.video import PushVideoLogger


def test_periodic_and_partial_clips(tmp_path, monkeypatch):
    logged = []
    uploaded = []

    def video(path, format):
        # Read the finished MP4 at upload time, including its encoded RGB pixels.
        capture = cv2.VideoCapture(path)
        uploaded.append((path, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        assert int(capture.get(cv2.CAP_PROP_FOURCC)) == cv2.VideoWriter_fourcc(*"h264")
        ok, frame = capture.read()
        capture.release()
        assert ok and frame.shape == (32, 48, 3)
        assert frame[..., 2].mean() > 240
        assert frame[..., :2].max() < 10
        assert format == "mp4"
        return path

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        run=SimpleNamespace(log=logged.append), Video=video,
    ))
    updates = []
    rgb = torch.zeros((4, 32, 48, 3), dtype=torch.uint8)
    rgb[..., 0] = 255
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            video_interval=10, video_length=0, video_every=2, video_envs=4,
            video_dir=str(tmp_path), log_dir=None,
            table_center=(0.55, 0., 0.), table_size=(0.8, 0.8, 0.05),
        ),
        num_envs=8, max_episode_length=6, step_dt=1 / 60, device="cpu",
        scene=SimpleNamespace(env_origins=torch.tensor([[2., 3., 0.]])),
        _recording_camera=SimpleNamespace(
            data=SimpleNamespace(output={"rgb": rgb}),
            update=lambda *args, **kwargs: updates.append((args, kwargs)),
        ),
    )
    recorder = PushVideoLogger(env)
    for _ in range(13):
        recorder.step()
    assert len(logged) == 1  # complete clip; the second remains open
    recorder.close()
    recorder.close()
    assert [count for _, count in uploaded] == [3] * 4 + [2] * 4
    assert len(list(tmp_path.glob("*.mp4"))) == 8
    assert len(logged) == 2
    assert [item["videos/env_steps"] for item in logged] == [48, 104]
    assert len(updates) == 5  # no camera reads between clips
