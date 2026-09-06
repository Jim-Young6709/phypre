"""
--------------------------------------------
Pure Codex generated scripts with no review.
--------------------------------------------

Load MolmoSpaces USD assets in an Isaac Lab standalone app.

Examples:
    ./IsaacLab/isaaclab.sh -p phy/load_molmospaces_isaaclab.py --steps 300

    ./IsaacLab/isaaclab.sh -p phy/load_molmospaces_isaaclab.py \
        --scene-name FloorPlan1_physics \
        --asset-id Apple_1

    ./IsaacLab/isaaclab.sh -p phy/load_molmospaces_isaaclab.py \
        --query "a 3D model of a cellphone"

    ./IsaacLab/isaaclab.sh -p phy/load_molmospaces_isaaclab.py \
        --scene /path/to/scenes/ithor/FloorPlan1_physics/scene.usda \
        --asset /path/to/objects/thor/20260128/Apple_1_mesh/Apple_1_mesh.usda
"""

from __future__ import annotations

# Launch Isaac Sim before importing Isaac/Omniverse modules.

import argparse
import json
import math
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
LOCAL_USD_ROOT = ROOT / "molmospaces" / "molmo_spaces_isaac" / "assets" / "usd"
CACHE_USD_ROOT = Path.home() / ".molmospaces" / "usd"
DEFAULT_SCENE_NAME = "FloorPlan1_physics"
DEFAULT_ASSET_ID = "Apple_1"
DEFAULT_OBJECT_POSITION = (0.0, -1.0, 0.0)
ASSET_AXIS_CONVENTIONS = ("y_up_to_z_up", "usd")
IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)
Y_UP_TO_Z_UP_WXYZ = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)


parser = argparse.ArgumentParser(description="Load MolmoSpaces thor/ithor USDs in Isaac Lab.")
parser.add_argument("--usd-root", type=Path, default=None, help="Root containing objects/ and scenes/.")
parser.add_argument("--scene", type=Path, default=None, help="Direct path to an ithor scene.usda.")
parser.add_argument("--scene-name", default=DEFAULT_SCENE_NAME, help="Scene folder name, e.g. FloorPlan1_physics.")
parser.add_argument("--asset", type=Path, default=None, help="Direct path to a thor *_mesh.usda or *_prim.usda.")
parser.add_argument("--asset-id", default=DEFAULT_ASSET_ID, help="THOR asset id, e.g. Apple_1 or Fridge_1.")
parser.add_argument("--query", default="", help="Search MolmoSpaces metadata and load the first local USD hit.")
parser.add_argument("--query-results", type=int, default=10, help="Number of ObjectRetriever hits to inspect.")
parser.add_argument("--query-threshold", type=float, default=0.5, help="Minimum ObjectRetriever similarity.")
parser.add_argument("--no-scene", action="store_true", help="Do not load an ithor scene.")
parser.add_argument("--no-asset", action="store_true", help="Do not load a separate thor object.")
parser.add_argument("--asset-position", type=float, nargs=3, default=DEFAULT_OBJECT_POSITION)
parser.add_argument(
    "--asset-axis-convention",
    choices=ASSET_AXIS_CONVENTIONS,
    default="y_up_to_z_up",
    help="THOR object source axis convention. Use `usd` to load the USD root transform without conversion.",
)
parser.add_argument("--steps", type=int, default=0, help="0 runs until the app closes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Everything below can use Isaac/Omniverse modules.

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402


def candidate_usd_roots() -> list[Path]:
    if args_cli.usd_root is not None:
        return [args_cli.usd_root.expanduser()]
    return [
        LOCAL_USD_ROOT,
        CACHE_USD_ROOT,
        ROOT / "assets" / "usd",
        Path.cwd() / "assets" / "usd",
    ]


def normalize_scene_name(scene_name: str) -> list[str]:
    names = [scene_name]
    if not scene_name.endswith("_physics"):
        names.append(f"{scene_name}_physics")
    return names


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def resolve_scene_path() -> Path | None:
    if args_cli.scene is not None:
        scene_path = args_cli.scene.expanduser()
        if not scene_path.is_file():
            raise FileNotFoundError(f"Scene USD was not found: {scene_path}")
        return scene_path

    for root in candidate_usd_roots():
        scene_candidates: list[Path] = []
        for scene_name in normalize_scene_name(args_cli.scene_name):
            scene_candidates.extend(
                [
                    root / "scenes" / "ithor" / scene_name / "scene.usda",
                    *sorted((root / "scenes" / "ithor").glob(f"*/{scene_name}/scene.usda")),
                ]
            )
        if scene_path := first_existing(scene_candidates):
            return scene_path

    for root in candidate_usd_roots():
        matches = sorted((root / "scenes" / "ithor").glob("**/scene.usda"))
        if matches:
            return matches[0]
    return None


def resolve_asset_id_path(asset_id: str) -> Path | None:
    for root in candidate_usd_roots():
        object_root = root / "objects" / "thor"
        asset_candidates = [
            object_root / f"{asset_id}_mesh" / f"{asset_id}_mesh.usda",
            object_root / f"{asset_id}_prim" / f"{asset_id}_prim.usda",
            *sorted(object_root.glob(f"*/{asset_id}_mesh/{asset_id}_mesh.usda")),
            *sorted(object_root.glob(f"*/{asset_id}_prim/{asset_id}_prim.usda")),
        ]
        if asset_path := first_existing(asset_candidates):
            return asset_path


def query_asset_ids() -> list[str]:
    from molmo_spaces.utils.object_metadata import ObjectMeta
    from molmo_spaces.utils.object_retriever import ObjectRetriever

    retriever = ObjectRetriever(sim_thres=args_cli.query_threshold, max_results=args_cli.query_results)
    uids, sims = retriever.query(args_cli.query)
    if len(uids) == 0:
        raise ValueError(f"ObjectRetriever returned no results for query: {args_cli.query!r}")

    print(f"[INFO] ObjectRetriever results for query: {args_cli.query!r}")
    asset_ids: list[str] = []
    for it, (uid, sim) in enumerate(zip(uids, sims, strict=True)):
        asset_id = str(uid)
        anno = ObjectMeta.annotation(asset_id) or {}
        desc_short = anno.get("description_short") or {}
        if isinstance(desc_short, dict):
            desc = desc_short.get("five_words") or desc_short.get("three_words") or desc_short.get("one_word") or ""
        else:
            desc = str(desc_short)
        print(
            f"  {it:02d} sim={float(sim):.3f} uid={asset_id} "
            f"obja={anno.get('isObjaverse', '?')} split={anno.get('split', '?')} "
            f"cat=`{anno.get('category', '?')}`: {desc}"
        )
        asset_ids.append(asset_id)
    return asset_ids


def resolve_asset_path() -> Path | None:
    if args_cli.asset is not None:
        asset_path = args_cli.asset.expanduser()
        if not asset_path.is_file():
            raise FileNotFoundError(f"Object USD was not found: {asset_path}")
        return asset_path

    if args_cli.query:
        for asset_id in query_asset_ids():
            asset_path = resolve_asset_id_path(asset_id)
            if asset_path is not None:
                print(f"[INFO] Using first local USD hit from query: {asset_id}")
                return asset_path
        raise FileNotFoundError(
            "ObjectRetriever found assets, but none of those IDs had a local thor USD file. "
            "Install matching USD objects or pass --asset/--asset-id explicitly."
        )

    if asset_path := resolve_asset_id_path(args_cli.asset_id):
        return asset_path

    for root in candidate_usd_roots():
        matches = sorted((root / "objects" / "thor").glob("**/*_mesh.usda"))
        if matches:
            return matches[0]
    return None


def asset_id_from_path(asset_path: Path) -> str:
    stem = asset_path.stem
    return stem.removesuffix("_mesh").removesuffix("_prim")


def axis_conversion_wxyz(convention: str) -> tuple[float, float, float, float]:
    if convention == "y_up_to_z_up":
        return Y_UP_TO_Z_UP_WXYZ
    if convention == "usd":
        return IDENTITY_WXYZ
    raise ValueError(f"Unknown asset axis convention {convention!r}.")


def metadata_paths() -> list[Path]:
    return [
        ROOT
        / "molmospaces"
        / "molmo_spaces_isaac"
        / "src"
        / "molmo_spaces_isaac"
        / "resources"
        / "usd_assets_metadata.json",
        ROOT / "molmospaces" / "molmo_spaces_isaac" / "usd_assets_metadata.json",
    ]


def load_bbox_height(asset_path: Path) -> float:
    asset_id = asset_id_from_path(asset_path)
    height_axis = 1 if args_cli.asset_axis_convention == "y_up_to_z_up" else 2
    for metadata_path in metadata_paths():
        if not metadata_path.is_file():
            continue
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata: dict[str, Any] = json.load(file)
        asset_metadata = metadata.get(asset_id)
        if asset_metadata is None:
            continue
        bbox_size = asset_metadata.get("bbox_size")
        if isinstance(bbox_size, list) and len(bbox_size) >= 3:
            return float(abs(bbox_size[height_axis]))
    return 0.20


def spawn_usd(prim_path: str, usd_path: Path, translation=None, orientation=None) -> None:
    cfg = sim_utils.UsdFileCfg(usd_path=usd_path.resolve().as_posix())
    cfg.func(prim_path, cfg, translation=translation, orientation=orientation)


def setup_scene() -> None:
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    if not args_cli.no_scene:
        scene_path = resolve_scene_path()
        if scene_path is None:
            raise FileNotFoundError("Could not find an ithor scene.usda under any known USD root.")
        spawn_usd("/World/MolmoIThorScene", scene_path)
        print(f"[INFO] Loaded ithor scene: {scene_path}")
    else:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    if not args_cli.no_asset:
        asset_path = resolve_asset_path()
        if asset_path is None:
            raise FileNotFoundError("Could not find a thor object USDA under any known USD root.")
        x, y, z_offset = args_cli.asset_position
        z = z_offset + load_bbox_height(asset_path) * 0.5 + 0.01
        orientation = axis_conversion_wxyz(args_cli.asset_axis_convention)
        spawn_usd("/World/MolmoThorObject", asset_path, translation=(x, y, z), orientation=orientation)
        print(f"[INFO] Loaded thor object: {asset_path} axis={args_cli.asset_axis_convention}")


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0))
    sim.set_camera_view(eye=[3.0, -4.0, 2.2], target=[0.0, 0.0, 0.6])

    setup_scene()
    sim.reset()

    print("[INFO] MolmoSpaces IsaacLab example is running.")
    count = 0
    while simulation_app.is_running() and (args_cli.steps <= 0 or count < args_cli.steps):
        sim.step()
        count += 1
    print(f"[INFO] Finished {count} simulation steps.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
