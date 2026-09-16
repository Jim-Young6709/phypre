"""Check local-file bypass and remote validation without launching Isaac Sim."""

import sys
from types import ModuleType, SimpleNamespace

from phy.utils.assets import enable_fast_usd_checks


def test_usd_checks_preserve_missing_files_and_remote_timeouts(tmp_path, monkeypatch):
    calls = []

    async def remote_check(path, timeout):
        calls.append((path, timeout))
        return path.startswith("https://")

    assets = SimpleNamespace(_is_usd_path_available=remote_check)
    spawner = SimpleNamespace()
    for name, attributes in (
        ("isaaclab.utils", {"assets": assets}),
        ("isaaclab.sim.spawners.from_files", {"from_files": spawner}),
    ):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    enable_fast_usd_checks()
    check = spawner.check_usd_path_with_timeout
    assert check is assets.check_usd_path_with_timeout
    local = tmp_path / "object.usda"
    local.write_text("#usda 1.0\n")
    assert check(str(local))
    assert calls == []
    missing = str(tmp_path / "missing.usda")
    assert not check(missing, timeout=2)
    assert check("https://example.com/robot.usd", timeout=7)
    assert calls == [(missing, 2), ("https://example.com/robot.usd", 7)]
