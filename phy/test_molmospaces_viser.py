#!/usr/bin/env python3
"""Visual-only Viser viewer for MolmoSpaces USD scenes and objects.

This viewer extracts triangle meshes from the composed USD stage and sends them
to Viser. It does not run Isaac physics or preserve USD materials/textures.

Examples:
    python phy/view_molmospaces_viser.py

    python phy/view_molmospaces_viser.py \
        --scene-name FloorPlan1_physics \
        --asset-id Apple_1

    python phy/view_molmospaces_viser.py \
        --scene /path/to/scenes/ithor/FloorPlan1_physics/scene.usda \
        --object /path/to/objects/thor/20260128/Apple_1_mesh/Apple_1_mesh.usda
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import viser
from molmo_spaces.utils.grasps import get_pickup_grasp_path
from molmo_spaces.utils.object_metadata import ObjectMeta
from molmo_spaces.utils.object_retriever import ObjectRetriever
from pxr import Gf, Usd, UsdGeom


ROOT = Path(__file__).resolve().parents[1]
LOCAL_USD_ROOT = ROOT / "molmospaces" / "molmo_spaces_isaac" / "assets" / "usd"
CACHE_USD_ROOT = Path.home() / ".molmospaces" / "usd"
DEFAULT_SCENE_NAME = "FloorPlan1_physics"
DEFAULT_ASSET_ID = "Apple_1"
DEFAULT_OBJECT_POSITION = (0.0, -1.0, 0.0)
ASSET_AXIS_CONVENTIONS = ("thor_y_up", "usd")
THOR_Y_UP_TO_Z_UP_MATRIX = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    mesh_count: int
    triangle_count: int

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.vertices) == 0:
            zeros = np.zeros(3, dtype=np.float32)
            return zeros, zeros
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


@dataclass(frozen=True)
class GraspFileData:
    label: str
    path: Path
    transforms: np.ndarray


@dataclass(frozen=True)
class SceneAssetInstance:
    asset_id: str
    prim_path: str
    world_from_asset: np.ndarray
    named_world_transforms: tuple[tuple[str, np.ndarray], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View MolmoSpaces ithor scenes and thor objects in Viser.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--usd-root", type=Path, default=None, help="Root containing objects/ and scenes/.")
    parser.add_argument("--scene", type=Path, default=None, help="Direct path to an ithor scene.usda.")
    parser.add_argument("--scene-name", default=DEFAULT_SCENE_NAME)
    parser.add_argument("--object", type=Path, default=None, help="Direct path to a thor *_mesh.usda or *_prim.usda.")
    parser.add_argument("--asset-id", default=DEFAULT_ASSET_ID)
    parser.add_argument("--object-position", type=float, nargs=3, default=DEFAULT_OBJECT_POSITION)
    parser.add_argument(
        "--asset-axis-convention",
        choices=ASSET_AXIS_CONVENTIONS,
        default="thor_y_up",
        help="THOR object source axis convention. Use `usd` to view the USD root transform without conversion.",
    )
    parser.add_argument("--max-scene-triangles", type=int, default=800_000)
    parser.add_argument("--max-object-triangles", type=int, default=200_000)
    parser.add_argument("--max-object-options", type=int, default=500)
    parser.add_argument("--include-collision", action="store_true")
    parser.add_argument("--no-scene", action="store_true")
    parser.add_argument("--no-object", action="store_true")
    parser.add_argument("--hide-grasps", action="store_true", help="Do not load grasp pose frames at startup.")
    parser.add_argument("--grasp-root", type=Path, default=None, help="Root containing grasps/, or the grasps root itself.")
    parser.add_argument("--grasp-source", default="droid", help="MolmoSpaces grasp library to use for thor assets.")
    parser.add_argument("--num-grasps", type=int, default=30, help="Grasps per file. Use 0 or negative to load all.")
    parser.add_argument(
        "--max-scene-grasp-frames",
        type=int,
        default=1500,
        help="Max scene grasp frames to draw. Use 0 or negative for no scene-wide cap.",
    )
    parser.add_argument("--include-flipped-grasps", action="store_true")
    parser.add_argument("--grasp-axis-length", type=float, default=0.06)
    return parser.parse_args()


def candidate_usd_roots(args: argparse.Namespace) -> list[Path]:
    if args.usd_root is not None:
        return [args.usd_root.expanduser()]
    return [
        LOCAL_USD_ROOT,
        CACHE_USD_ROOT,
        ROOT / "assets" / "usd",
        Path.cwd() / "assets" / "usd",
    ]


def asset_id_from_path(path: Path) -> str:
    return path.stem.removesuffix("_mesh").removesuffix("_prim")


def discover_scenes(args: argparse.Namespace) -> dict[str, Path]:
    scenes: dict[str, Path] = {}
    if args.scene is not None:
        path = args.scene.expanduser().absolute()
        if path.is_file():
            scenes[path.parent.name] = path
    for root in candidate_usd_roots(args):
        scene_root = root / "scenes" / "ithor"
        for path in sorted(scene_root.glob("**/scene.usda")):
            scenes.setdefault(path.parent.name, path.absolute())
    return scenes


def discover_objects(args: argparse.Namespace) -> dict[str, Path]:
    objects: dict[str, Path] = {}
    if args.object is not None:
        path = args.object.expanduser().absolute()
        if path.is_file():
            objects[asset_id_from_path(path)] = path
    for root in candidate_usd_roots(args):
        object_root = root / "objects" / "thor"
        for pattern in ("**/*_mesh.usda", "**/*_prim.usda"):
            for path in sorted(object_root.glob(pattern)):
                asset_id = asset_id_from_path(path)
                old_path = objects.get(asset_id)
                if old_path is None or old_path.stem.endswith("_prim"):
                    objects[asset_id] = path.absolute()
    return objects


def normalize_scene_name(scene_name: str) -> list[str]:
    names = [scene_name]
    if not scene_name.endswith("_physics"):
        names.append(f"{scene_name}_physics")
    return names


def initial_scene_name(args: argparse.Namespace, scenes: dict[str, Path]) -> str:
    for scene_name in normalize_scene_name(args.scene_name):
        if scene_name in scenes:
            return scene_name
    if not scenes:
        return "(none)"
    return sorted(scenes)[0]


def initial_object_name(args: argparse.Namespace, objects: dict[str, Path]) -> str:
    if args.asset_id in objects:
        return args.asset_id
    if not objects:
        return "(none)"
    return sorted(objects)[0]


def is_visual_mesh(prim: Usd.Prim, include_collision: bool) -> bool:
    if not prim.IsA(UsdGeom.Mesh):
        return False
    path = str(prim.GetPath()).lower()
    if not include_collision and ("collision" in path or "collider" in path):
        return False
    imageable = UsdGeom.Imageable(prim)
    if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
        return False
    purpose = imageable.ComputePurpose()
    if not include_collision and purpose in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
        return False
    return True


def points_to_numpy(points_attr_value) -> np.ndarray:
    return np.asarray([(p[0], p[1], p[2]) for p in points_attr_value], dtype=np.float64)


def transform_points(points: np.ndarray, matrix: Gf.Matrix4d) -> np.ndarray:
    out = np.empty_like(points, dtype=np.float64)
    for idx, point in enumerate(points):
        transformed = matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        out[idx] = (transformed[0], transformed[1], transformed[2])
    return out


def triangulate_faces(counts, indices, max_triangles: int) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        face_indices = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        root = int(face_indices[0])
        for offset in range(1, count - 1):
            faces.append((root, int(face_indices[offset]), int(face_indices[offset + 1])))
            if len(faces) >= max_triangles:
                return np.asarray(faces, dtype=np.uint32)
    return np.asarray(faces, dtype=np.uint32)


@lru_cache(maxsize=16)
def extract_usd_mesh(path_str: str, include_collision: bool, max_triangles: int) -> MeshData:
    stage = Usd.Stage.Open(path_str)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path_str}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertex_chunks: list[np.ndarray] = []
    face_chunks: list[np.ndarray] = []
    vertex_offset = 0
    total_triangles = 0
    mesh_count = 0

    for prim in stage.Traverse():
        if not is_visual_mesh(prim, include_collision):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_value = mesh.GetPointsAttr().Get()
        counts_value = mesh.GetFaceVertexCountsAttr().Get()
        indices_value = mesh.GetFaceVertexIndicesAttr().Get()
        if points_value is None or counts_value is None or indices_value is None:
            continue
        if len(points_value) == 0 or len(counts_value) == 0 or len(indices_value) == 0:
            continue

        remaining = max_triangles - total_triangles
        if remaining <= 0:
            break

        points = points_to_numpy(points_value)
        if len(points) == 0:
            continue
        faces = triangulate_faces(list(counts_value), list(indices_value), remaining)
        if len(faces) == 0:
            continue

        world_points = transform_points(points, xform_cache.GetLocalToWorldTransform(prim))
        vertex_chunks.append(world_points.astype(np.float32))
        face_chunks.append((faces + vertex_offset).astype(np.uint32))
        vertex_offset += len(points)
        total_triangles += len(faces)
        mesh_count += 1

    if not vertex_chunks or not face_chunks:
        raise RuntimeError(f"No visual meshes were extracted from: {path_str}")

    return MeshData(
        vertices=np.concatenate(vertex_chunks, axis=0),
        faces=np.concatenate(face_chunks, axis=0),
        mesh_count=mesh_count,
        triangle_count=total_triangles,
    )


def axis_conversion_matrix(convention: str) -> np.ndarray:
    if convention == "thor_y_up":
        return THOR_Y_UP_TO_Z_UP_MATRIX
    if convention == "usd":
        return np.eye(3, dtype=np.float64)
    raise ValueError(f"Unknown asset axis convention {convention!r}.")


def converted_object_vertices(mesh: MeshData, convention: str) -> np.ndarray:
    return mesh.vertices.astype(np.float64) @ axis_conversion_matrix(convention).T


def object_visual_transform(mesh: MeshData, position: tuple[float, float, float], convention: str) -> np.ndarray:
    rotation = axis_conversion_matrix(convention)
    vertices = converted_object_vertices(mesh, convention)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(
        [
            float(position[0]) - float(center_xy[0]),
            float(position[1]) - float(center_xy[1]),
            float(position[2]) - float(bounds_min[2]),
        ],
        dtype=np.float64,
    )
    return transform


def grounded_object_mesh(mesh: MeshData, position: tuple[float, float, float], convention: str) -> tuple[MeshData, np.ndarray]:
    vertices = converted_object_vertices(mesh, convention).astype(np.float32)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2])
    vertices[:, 0] -= center_xy[0]
    vertices[:, 1] -= center_xy[1]
    vertices[:, 2] -= bounds_min[2]
    vertices += np.asarray(position, dtype=np.float32)
    return (
        MeshData(vertices, mesh.faces.copy(), mesh.mesh_count, mesh.triangle_count),
        object_visual_transform(mesh, position, convention),
    )


def matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        diag_idx = int(np.argmax(np.diag(matrix)))
        if diag_idx == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif diag_idx == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def flip_grasp_transforms(transforms: np.ndarray) -> np.ndarray:
    flip_z = np.eye(4, dtype=np.float64)
    flip_z[:3, :3] = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transforms @ flip_z


def candidate_grasp_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    if args.grasp_root is not None:
        roots.append(args.grasp_root.expanduser())
    for env_name in ("MLSPACES_ASSETS_DIR", "MLSPACES_CACHE_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            roots.append(Path(env_value).expanduser())
    roots.extend(
        [
            ROOT / "molmospaces" / "examples" / "custom_assets" / "asset_library",
            Path.home() / ".cache" / "molmospaces" / "assets",
            Path.home() / ".cache" / "molmo-spaces-resources",
        ]
    )

    expanded_roots: list[Path] = []
    for root in roots:
        expanded_roots.append(root)
        if root.name == "assets" and root.is_dir():
            expanded_roots.extend(child for child in root.iterdir() if child.is_dir())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in expanded_roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in seen:
            deduped.append(root)
            seen.add(resolved)
    return deduped


def dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            key = path.resolve()
        except OSError:
            key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def safe_frame_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "unnamed"


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def gf_matrix_to_numpy(matrix: Gf.Matrix4d) -> np.ndarray:
    # USD/Gf prints translation in the last row; Viser/numpy composition below uses column-vector convention.
    return np.asarray(matrix, dtype=np.float64).T.copy()


def asset_id_from_reference_path(asset_path: str) -> str | None:
    normalized = asset_path.replace("\\", "/")
    if "/objects/thor/" not in normalized and not normalized.startswith("objects/thor/"):
        return None
    return asset_id_from_path(Path(normalized))


def iter_reference_asset_paths(prim: Usd.Prim) -> list[str]:
    refs = prim.GetMetadata("references")
    if refs is None:
        return []

    items = []
    try:
        items.extend(refs.GetAddedOrExplicitItems())
    except Exception:
        pass
    for attr_name in ("prependedItems", "appendedItems", "explicitItems"):
        items.extend(getattr(refs, attr_name, []) or [])

    paths: list[str] = []
    seen: set[str] = set()
    for ref in items:
        asset_path = str(getattr(ref, "assetPath", "") or "")
        if asset_path and asset_path not in seen:
            paths.append(asset_path)
            seen.add(asset_path)
    return paths


def skip_transform_name(name: str) -> bool:
    lowered = name.lower()
    return "collision" in lowered or "collider" in lowered


def build_named_world_transforms(root_prim: Usd.Prim, xform_cache: UsdGeom.XformCache) -> tuple[tuple[str, np.ndarray], ...]:
    transforms: list[tuple[str, np.ndarray]] = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Xformable):
            continue
        name = prim.GetName()
        if not name or skip_transform_name(name):
            continue
        try:
            transform = gf_matrix_to_numpy(xform_cache.GetLocalToWorldTransform(prim))
        except Exception:
            continue
        transforms.append((name, transform))
    return tuple(transforms)


@lru_cache(maxsize=16)
def discover_scene_asset_instances(path_str: str) -> tuple[SceneAssetInstance, ...]:
    stage = Usd.Stage.Open(path_str)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path_str}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    instances: list[SceneAssetInstance] = []
    seen_prim_paths: set[str] = set()
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path in seen_prim_paths:
            continue
        asset_id = None
        for asset_path in iter_reference_asset_paths(prim):
            asset_id = asset_id_from_reference_path(asset_path)
            if asset_id is not None:
                break
        if asset_id is None:
            continue

        try:
            world_from_asset = gf_matrix_to_numpy(xform_cache.GetLocalToWorldTransform(prim))
        except Exception:
            world_from_asset = np.eye(4, dtype=np.float64)
        instances.append(
            SceneAssetInstance(
                asset_id=asset_id,
                prim_path=prim_path,
                world_from_asset=world_from_asset,
                named_world_transforms=build_named_world_transforms(prim, xform_cache),
            )
        )
        seen_prim_paths.add(prim_path)
    return tuple(instances)


@lru_cache(maxsize=128)
def discover_object_asset_frames(path_str: str) -> tuple[tuple[str, np.ndarray], ...]:
    stage = Usd.Stage.Open(path_str)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path_str}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    transforms: list[tuple[str, np.ndarray]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        name = prim.GetName()
        if not name or skip_transform_name(name):
            continue
        try:
            transforms.append((name, gf_matrix_to_numpy(xform_cache.GetLocalToWorldTransform(prim))))
        except Exception:
            continue
    return tuple(transforms)


def grasp_dir_candidates(asset_id: str, source: str, root: Path) -> list[Path]:
    candidates = [
        root / "grasps" / source / asset_id,
        root / source / asset_id,
        root / asset_id,
    ]
    for pattern in (f"grasps/{source}/*/{asset_id}", f"{source}/*/{asset_id}"):
        try:
            candidates.extend(path for path in root.glob(pattern) if path.is_dir())
        except OSError:
            continue
    return dedupe_paths(candidates)


def grasp_label_from_path(path: Path, asset_id: str) -> str:
    if path.name == "grasps.npz":
        return asset_id
    return path.stem.removesuffix("_grasps_filtered")


def list_asset_grasp_files(asset_id: str, args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for root in candidate_grasp_roots(args):
        for grasp_dir in grasp_dir_candidates(asset_id, args.grasp_source, root):
            if not grasp_dir.is_dir():
                continue
            files.extend(sorted(grasp_dir.glob("*_grasps_filtered.npz")))
            custom_path = grasp_dir / "droid" / "grasps.npz"
            if custom_path.is_file():
                files.append(custom_path)

    try:
        pickup_path = Path(get_pickup_grasp_path(asset_id, grasp_libraries=[args.grasp_source]))
        if pickup_path.is_file():
            files.append(pickup_path)
    except Exception:
        pass

    def sort_key(path: Path) -> tuple[int, str]:
        is_pickup = path.name in {f"{asset_id}_grasps_filtered.npz", "grasps.npz"}
        return (0 if is_pickup else 1, path.name)

    return sorted(dedupe_paths(files), key=sort_key)


def load_npz_transforms(path: Path, args: argparse.Namespace) -> np.ndarray:
    with np.load(path) as data:
        if "transforms" not in data:
            keys = ", ".join(sorted(data.files))
            raise KeyError(f"`{path}` has no `transforms` array. Available keys: {keys}")
        transforms = np.asarray(data["transforms"], dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Grasp file `{path}` has transforms shape {transforms.shape}, expected (N, 4, 4).")
    if int(args.num_grasps) > 0 and len(transforms) > int(args.num_grasps):
        transforms = transforms[: int(args.num_grasps)]
    if args.include_flipped_grasps:
        transforms = np.concatenate([transforms, flip_grasp_transforms(transforms)], axis=0)
    return transforms


def load_asset_grasp_sets(asset_id: str, args: argparse.Namespace) -> list[GraspFileData]:
    files = list_asset_grasp_files(asset_id, args)
    if not files:
        raise FileNotFoundError(
            f"No grasp files found for `{asset_id}`. Expected files under "
            f"`grasps/{args.grasp_source}/<version-or-current>/{asset_id}/` in MLSPACES_ASSETS_DIR, "
            "MLSPACES_CACHE_DIR, or --grasp-root."
        )

    grasp_sets: list[GraspFileData] = []
    errors: list[str] = []
    for path in files:
        try:
            transforms = load_npz_transforms(path, args)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if len(transforms) == 0:
            continue
        grasp_sets.append(GraspFileData(grasp_label_from_path(path, asset_id), path, transforms))

    if not grasp_sets:
        detail = f" Errors: {'; '.join(errors[:3])}" if errors else ""
        raise RuntimeError(f"Found grasp files for `{asset_id}`, but none loaded usable transforms.{detail}")
    return grasp_sets


@lru_cache(maxsize=256)
def joint_frame_preferences(grasp_dir_str: str) -> dict[str, tuple[str, ...]]:
    info_path = Path(grasp_dir_str) / "joint_meshes_info_filtered.json"
    if not info_path.is_file():
        return {}
    try:
        raw = json.loads(info_path.read_text())
    except Exception:
        return {}

    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = raw.get("joint_meshes_info") or raw.get("joints") or list(raw.values())
    else:
        rows = []

    mapping: dict[str, tuple[str, ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        joint_info = row.get("joint_info")
        if not isinstance(joint_info, dict):
            joint_info = {}
        file_name = str(row.get("filtered_grasps_file") or "")
        if not file_name and row.get("joint"):
            file_name = f"{row['joint']}_grasps_filtered.npz"
        if not file_name:
            continue

        names: list[str] = []
        parent_body = row.get("parent_body") or joint_info.get("parent_body")
        if isinstance(parent_body, str) and parent_body:
            names.append(parent_body)
        handle_geoms = row.get("handle_geoms") or []
        if isinstance(handle_geoms, str):
            handle_geoms = [handle_geoms]
        names.extend(str(value) for value in handle_geoms if value)
        for key in ("joint", "name"):
            value = row.get(key) or joint_info.get(key)
            if isinstance(value, str) and value:
                names.append(value)
        mapping[file_name] = tuple(dict.fromkeys(names))
    return mapping


def match_named_transform(
    named_transforms: tuple[tuple[str, np.ndarray], ...],
    preferred_names: tuple[str, ...],
    fallback_label: str,
) -> tuple[np.ndarray | None, str | None]:
    by_normalized = {normalized_name(name): (name, transform) for name, transform in named_transforms}
    for preferred_name in preferred_names:
        match = by_normalized.get(normalized_name(preferred_name))
        if match is not None:
            return match[1], match[0]

    label_norm = normalized_name(fallback_label)
    if not label_norm:
        return None, None

    best: tuple[int, str, np.ndarray] | None = None
    for name, transform in named_transforms:
        name_norm = normalized_name(name)
        if not name_norm:
            continue
        if name_norm == label_norm:
            score = 10_000 + len(name_norm)
        elif label_norm in name_norm:
            score = len(label_norm)
        elif name_norm in label_norm:
            score = len(name_norm)
        else:
            continue
        if best is None or score > best[0]:
            best = (score, name, transform)
    if best is None:
        return None, None
    return best[2], best[1]


def select_grasp_frame(instance: SceneAssetInstance, grasp_file: Path) -> tuple[np.ndarray, str]:
    if grasp_file.name == f"{instance.asset_id}_grasps_filtered.npz" or grasp_file.name == "grasps.npz":
        return instance.world_from_asset, "asset"

    preferences = joint_frame_preferences(grasp_file.parent.as_posix()).get(grasp_file.name, ())
    label = grasp_file.stem.removesuffix("_grasps_filtered")
    transform, frame_name = match_named_transform(instance.named_world_transforms, preferences, label)
    if transform is not None and frame_name is not None:
        return transform, frame_name
    return instance.world_from_asset, "asset"


def bbox_edges(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    bounds_min, bounds_max = mesh.bounds
    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    corners = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edge_indices = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int64,
    )
    edges = corners[edge_indices]
    colors = np.tile(np.asarray((255, 190, 40), dtype=np.uint8), (len(edges), 2, 1))
    return edges, colors


class MolmoViserViewer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.scenes = discover_scenes(args)
        self.objects = discover_objects(args)
        if not self.scenes and not args.no_scene:
            raise FileNotFoundError("No ithor scene.usda files found. Pass --scene or --usd-root.")
        if not self.objects and not args.no_object:
            raise FileNotFoundError("No thor object USDA files found. Pass --object or --usd-root.")

        self.scene_names = sorted(self.scenes)
        self.object_names = sorted(self.objects)
        self.current_scene_name = initial_scene_name(args, self.scenes)
        self.current_object_name = initial_object_name(args, self.objects)
        self.scene_handle = None
        self.object_handle = None
        self.object_bbox_handle = None
        self.scene_mesh: MeshData | None = None
        self.object_mesh: MeshData | None = None
        self.object_to_viser: np.ndarray | None = None
        self.loaded_scene_path: Path | None = None
        self.loaded_object_name: str | None = None
        self.loaded_object_path: Path | None = None
        self.grasp_handles: list[object] = []
        self.grasp_path: str | None = None
        self.grasp_error: str | None = None
        self.scene_grasp_handles: list[object] = []
        self.scene_grasp_error: str | None = None

        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.gui.configure_theme(control_width="large")
        self.server.scene.add_frame("/axes", show_axes=True, axes_length=0.25, axes_radius=0.01)
        self.server.scene.add_grid("/grid", width=8.0, height=8.0, cell_size=0.25, section_size=1.0, shadow_opacity=0.2)

        self._build_gui()

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self.frame_client(client)

        if not args.no_scene and self.current_scene_name != "(none)":
            self.load_scene(self.current_scene_name)
        if not args.no_object and self.current_object_name != "(none)":
            self.load_object(self.current_object_name)

    def _build_gui(self) -> None:
        self.status_md = self.server.gui.add_markdown("Loading MolmoSpaces viewer...")

        with self.server.gui.add_folder("Scene"):
            scene_options = self.scene_names or ["(none)"]
            self.scene_dropdown = self.server.gui.add_dropdown("Scene", scene_options, initial_value=self.current_scene_name)
            self.scene_text = self.server.gui.add_text("Scene name", self.current_scene_name)
            self.load_scene_btn = self.server.gui.add_button("Load Scene")
            self.prev_scene_btn = self.server.gui.add_button("Prev Scene")
            self.next_scene_btn = self.server.gui.add_button("Next Scene")

        object_options = self.object_names[: self.args.max_object_options] or ["(none)"]
        if self.current_object_name not in object_options and self.current_object_name != "(none)":
            object_options = [self.current_object_name, *object_options]
        with self.server.gui.add_folder("Object"):
            self.object_dropdown = self.server.gui.add_dropdown("Object", object_options, initial_value=self.current_object_name)
            self.object_text = self.server.gui.add_text("Object ID", self.current_object_name)
            self.object_query = self.server.gui.add_text("Object query", "")
            self.load_object_btn = self.server.gui.add_button("Load Object")
            self.search_object_btn = self.server.gui.add_button("Search + Load")
            self.prev_object_btn = self.server.gui.add_button("Prev Object")
            self.next_object_btn = self.server.gui.add_button("Next Object")

        with self.server.gui.add_folder("Grasps"):
            self.show_grasps = self.server.gui.add_checkbox("Show object grasps", initial_value=not self.args.hide_grasps)
            self.load_grasps_btn = self.server.gui.add_button("Reload Object Grasps")
            self.show_scene_grasps = self.server.gui.add_checkbox(
                "Show scene grasps",
                initial_value=not self.args.hide_grasps and not self.args.no_scene,
            )
            self.load_scene_grasps_btn = self.server.gui.add_button("Reload Scene Grasps")

        with self.server.gui.add_folder("View"):
            self.show_scene = self.server.gui.add_checkbox("Show scene", initial_value=not self.args.no_scene)
            self.show_object = self.server.gui.add_checkbox("Show object", initial_value=not self.args.no_object)
            self.show_bbox = self.server.gui.add_checkbox("Show object bbox", initial_value=True)
            self.adjust_view_btn = self.server.gui.add_button("Adjust View")

        @self.scene_dropdown.on_update
        def _(_: object) -> None:
            self.current_scene_name = str(self.scene_dropdown.value)
            self.scene_text.value = self.current_scene_name

        @self.object_dropdown.on_update
        def _(_: object) -> None:
            self.current_object_name = str(self.object_dropdown.value)
            self.object_text.value = self.current_object_name

        @self.load_scene_btn.on_click
        def _(_: object) -> None:
            name = str(self.scene_text.value).strip() or self.current_scene_name
            self.load_scene(name)

        @self.prev_scene_btn.on_click
        def _(_: object) -> None:
            self.step_scene(-1)

        @self.next_scene_btn.on_click
        def _(_: object) -> None:
            self.step_scene(1)

        @self.load_object_btn.on_click
        def _(_: object) -> None:
            name = str(self.object_text.value).strip() or self.current_object_name
            self.load_object(name)

        @self.search_object_btn.on_click
        def _(_: object) -> None:
            self.search_and_load_object(str(self.object_query.value).strip())

        @self.prev_object_btn.on_click
        def _(_: object) -> None:
            self.step_object(-1)

        @self.next_object_btn.on_click
        def _(_: object) -> None:
            self.step_object(1)

        @self.load_grasps_btn.on_click
        def _(_: object) -> None:
            self.load_grasps_for_current_object()

        @self.load_scene_grasps_btn.on_click
        def _(_: object) -> None:
            self.load_scene_grasps_for_current_scene()

        @self.show_grasps.on_update
        def _(_: object) -> None:
            if bool(self.show_grasps.value) and not self.grasp_handles and self.loaded_object_name is not None:
                self.load_grasps_for_current_object()
                return
            self.set_grasp_visibility()

        @self.show_scene_grasps.on_update
        def _(_: object) -> None:
            if bool(self.show_scene_grasps.value) and not self.scene_grasp_handles and self.loaded_scene_path is not None:
                self.load_scene_grasps_for_current_scene()
                return
            self.set_scene_grasp_visibility()

        @self.show_scene.on_update
        def _(_: object) -> None:
            if self.scene_handle is not None:
                self.scene_handle.visible = bool(self.show_scene.value)
            self.set_scene_grasp_visibility()

        @self.show_object.on_update
        def _(_: object) -> None:
            visible = bool(self.show_object.value)
            if self.object_handle is not None:
                self.object_handle.visible = visible
            if self.object_bbox_handle is not None:
                self.object_bbox_handle.visible = visible and bool(self.show_bbox.value)
            self.set_grasp_visibility()

        @self.show_bbox.on_update
        def _(_: object) -> None:
            if self.object_bbox_handle is not None:
                self.object_bbox_handle.visible = bool(self.show_bbox.value) and bool(self.show_object.value)

        @self.adjust_view_btn.on_click
        def _(_: object) -> None:
            self.frame_all_clients()

    def set_status(self, text: str) -> None:
        print(text)
        self.status_md.content = text

    def resolve_scene_name(self, name: str) -> str:
        for candidate in normalize_scene_name(name):
            if candidate in self.scenes:
                return candidate
        raise KeyError(f"Scene not found: {name}")

    def resolve_object_name(self, name: str) -> str:
        if name in self.objects:
            return name
        raise KeyError(f"Object not found: {name}")

    def remove_handle(self, attr_name: str) -> None:
        handle = getattr(self, attr_name)
        if handle is None:
            return
        try:
            handle.remove()
        finally:
            setattr(self, attr_name, None)

    def clear_grasps(self) -> None:
        for handle in self.grasp_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.grasp_handles = []
        self.grasp_path = None
        self.grasp_error = None

    def clear_scene_grasps(self) -> None:
        for handle in self.scene_grasp_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.scene_grasp_handles = []
        self.scene_grasp_error = None

    def set_grasp_visibility(self) -> None:
        visible = bool(self.show_grasps.value) and bool(self.show_object.value)
        for handle in self.grasp_handles:
            try:
                handle.visible = visible
            except Exception:
                pass

    def set_scene_grasp_visibility(self) -> None:
        visible = bool(self.show_scene_grasps.value) and bool(self.show_scene.value)
        for handle in self.scene_grasp_handles:
            try:
                handle.visible = visible
            except Exception:
                pass

    def add_grasp_frame(self, name: str, transform: np.ndarray, handles: list[object], axis_scale: float = 1.0) -> None:
        axis_length = float(self.args.grasp_axis_length) * axis_scale
        handle = self.server.scene.add_frame(
            name,
            position=transform[:3, 3],
            wxyz=matrix_to_wxyz(transform[:3, :3]),
            show_axes=True,
            axes_length=axis_length,
            axes_radius=max(0.002, axis_length * 0.035),
        )
        handles.append(handle)

    def current_object_instance(self) -> SceneAssetInstance | None:
        if self.loaded_object_name is None or self.loaded_object_path is None or self.object_to_viser is None:
            return None
        named_transforms = tuple(
            (name, self.object_to_viser @ transform)
            for name, transform in discover_object_asset_frames(self.loaded_object_path.as_posix())
        )
        return SceneAssetInstance(
            asset_id=self.loaded_object_name,
            prim_path=self.loaded_object_name,
            world_from_asset=self.object_to_viser,
            named_world_transforms=named_transforms,
        )

    def load_scene(self, name: str) -> None:
        try:
            scene_name = self.resolve_scene_name(name)
            path = self.scenes[scene_name]
            self.set_status(f"Loading scene `{scene_name}`...")
            mesh = extract_usd_mesh(path.as_posix(), self.args.include_collision, self.args.max_scene_triangles)
            self.remove_handle("scene_handle")
            self.clear_scene_grasps()
            self.scene_handle = self.server.scene.add_mesh_simple(
                "/molmo/scene",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=(185, 188, 194),
                opacity=0.95,
                cast_shadow=False,
                receive_shadow=True,
                flat_shading=False,
                side="double",
            )
            self.scene_handle.visible = bool(self.show_scene.value)
            self.scene_mesh = mesh
            self.loaded_scene_path = path
            self.current_scene_name = scene_name
            self.scene_text.value = scene_name
            if scene_name in self.scene_dropdown.options:
                self.scene_dropdown.value = scene_name
            if bool(self.show_scene_grasps.value):
                grasp_count = self.load_scene_grasps_for_current_scene(update_status=False)
                grasp_suffix = f", {grasp_count} scene grasps" if grasp_count else ", no scene grasps loaded"
            else:
                grasp_suffix = ""
            self.set_status(
                f"Loaded scene `{scene_name}` from `{path}` "
                f"({mesh.mesh_count} meshes, {mesh.triangle_count} triangles{grasp_suffix})."
            )
            self.frame_all_clients()
        except Exception as exc:
            self.loaded_scene_path = None
            self.set_status(f"Failed to load scene `{name}`: `{exc}`")

    def load_object(self, name: str) -> None:
        try:
            object_name = self.resolve_object_name(name)
            path = self.objects[object_name]
            self.set_status(f"Loading object `{object_name}`...")
            raw_mesh = extract_usd_mesh(path.as_posix(), self.args.include_collision, self.args.max_object_triangles)
            mesh, object_to_viser = grounded_object_mesh(
                raw_mesh,
                tuple(float(v) for v in self.args.object_position),
                self.args.asset_axis_convention,
            )

            self.remove_handle("object_handle")
            self.remove_handle("object_bbox_handle")
            self.clear_grasps()
            self.object_handle = self.server.scene.add_mesh_simple(
                "/molmo/object",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=(240, 135, 45),
                opacity=1.0,
                cast_shadow=False,
                receive_shadow=True,
                flat_shading=False,
                side="double",
            )
            edges, colors = bbox_edges(mesh)
            self.object_bbox_handle = self.server.scene.add_line_segments(
                "/molmo/object_bbox",
                points=edges,
                colors=colors,
                line_width=2.0,
            )
            visible = bool(self.show_object.value)
            self.object_handle.visible = visible
            self.object_bbox_handle.visible = visible and bool(self.show_bbox.value)
            self.object_mesh = mesh
            self.object_to_viser = object_to_viser
            self.loaded_object_name = object_name
            self.loaded_object_path = path
            self.current_object_name = object_name
            self.object_text.value = object_name
            if object_name in self.object_dropdown.options:
                self.object_dropdown.value = object_name
            if bool(self.show_grasps.value):
                grasp_count = self.load_grasps_for_current_object(update_status=False)
                grasp_suffix = f", {grasp_count} grasps" if grasp_count else ", no grasps loaded"
            else:
                grasp_suffix = ""
            self.set_status(
                f"Loaded object `{object_name}` from `{path}` "
                f"axis={self.args.asset_axis_convention} "
                f"({mesh.mesh_count} meshes, {mesh.triangle_count} triangles{grasp_suffix})."
            )
            self.frame_all_clients()
        except Exception as exc:
            self.loaded_object_path = None
            self.set_status(f"Failed to load object `{name}`: `{exc}`")

    def load_grasps_for_current_object(self, update_status: bool = True) -> int:
        object_name = self.loaded_object_name
        object_instance = self.current_object_instance()
        if object_name is None or object_instance is None:
            if update_status:
                self.set_status("Load an object before loading grasps.")
            return 0
        try:
            self.clear_grasps()
            grasp_sets = load_asset_grasp_sets(object_name, self.args)
            for file_idx, grasp_set in enumerate(grasp_sets):
                world_from_frame, frame_name = select_grasp_frame(object_instance, grasp_set.path)
                for pose_idx, grasp in enumerate(grasp_set.transforms):
                    viser_grasp = world_from_frame @ grasp
                    self.add_grasp_frame(
                        f"/molmo/object_grasps/{file_idx:03d}_{safe_frame_name(grasp_set.label)}_"
                        f"{safe_frame_name(frame_name)}/{pose_idx:03d}",
                        viser_grasp,
                        self.grasp_handles,
                    )
            self.grasp_path = f"{len(grasp_sets)} files"
            self.set_grasp_visibility()
            if update_status:
                self.set_status(
                    f"Loaded {len(self.grasp_handles)} grasp poses for `{object_name}` from {len(grasp_sets)} files."
                )
            return len(self.grasp_handles)
        except Exception as exc:
            self.clear_grasps()
            self.grasp_error = str(exc)
            if update_status:
                self.set_status(f"Failed to load grasps for `{object_name}`: `{exc}`")
            return 0

    def load_scene_grasps_for_current_scene(self, update_status: bool = True) -> int:
        if self.loaded_scene_path is None:
            if update_status:
                self.set_status("Load a scene before loading scene grasps.")
            return 0

        try:
            self.clear_scene_grasps()
            if update_status:
                self.set_status(f"Loading scene grasps for `{self.current_scene_name}`...")

            instances = discover_scene_asset_instances(self.loaded_scene_path.as_posix())
            max_frames = int(self.args.max_scene_grasp_frames)
            has_cap = max_frames > 0
            clipped = False
            instance_count = 0
            file_count = 0
            missing_count = 0
            error_count = 0

            for instance_idx, instance in enumerate(instances):
                if has_cap and len(self.scene_grasp_handles) >= max_frames:
                    clipped = True
                    break
                try:
                    grasp_sets = load_asset_grasp_sets(instance.asset_id, self.args)
                except FileNotFoundError:
                    missing_count += 1
                    continue
                except Exception:
                    error_count += 1
                    continue

                instance_used = False
                for file_idx, grasp_set in enumerate(grasp_sets):
                    if has_cap and len(self.scene_grasp_handles) >= max_frames:
                        clipped = True
                        break
                    world_from_frame, frame_name = select_grasp_frame(instance, grasp_set.path)
                    for pose_idx, grasp in enumerate(grasp_set.transforms):
                        if has_cap and len(self.scene_grasp_handles) >= max_frames:
                            clipped = True
                            break
                        world_grasp = world_from_frame @ grasp
                        self.add_grasp_frame(
                            f"/molmo/scene_grasps/{instance_idx:03d}_{safe_frame_name(instance.asset_id)}/"
                            f"{file_idx:03d}_{safe_frame_name(grasp_set.label)}_{safe_frame_name(frame_name)}/"
                            f"{pose_idx:03d}",
                            world_grasp,
                            self.scene_grasp_handles,
                            axis_scale=0.8,
                        )
                        instance_used = True
                    if clipped:
                        break
                    file_count += 1
                if instance_used:
                    instance_count += 1
                if clipped:
                    break

            self.set_scene_grasp_visibility()
            suffix = " clipped by --max-scene-grasp-frames" if clipped else ""
            if update_status:
                self.set_status(
                    f"Loaded {len(self.scene_grasp_handles)} scene grasp poses from {instance_count} "
                    f"object instances and {file_count} files{suffix}. "
                    f"Missing grasp dirs: {missing_count}; load errors: {error_count}."
                )
            return len(self.scene_grasp_handles)
        except Exception as exc:
            self.clear_scene_grasps()
            self.scene_grasp_error = str(exc)
            if update_status:
                self.set_status(f"Failed to load scene grasps for `{self.current_scene_name}`: `{exc}`")
            return 0

    def step_scene(self, direction: int) -> None:
        if not self.scene_names:
            return
        current = self.current_scene_name if self.current_scene_name in self.scene_names else self.scene_names[0]
        idx = (self.scene_names.index(current) + direction) % len(self.scene_names)
        self.load_scene(self.scene_names[idx])

    def step_object(self, direction: int) -> None:
        if not self.object_names:
            return
        current = self.current_object_name if self.current_object_name in self.object_names else self.object_names[0]
        idx = (self.object_names.index(current) + direction) % len(self.object_names)
        self.load_object(self.object_names[idx])

    def search_and_load_object(self, query: str) -> None:
        if not query:
            self.set_status("Enter an object query first.")
            return
        try:
            retriever = ObjectRetriever(max_results=25)
            uids, sims = retriever.query(query)
            if len(uids) == 0:
                self.set_status(f"No ObjectRetriever hits for `{query}`.")
                return

            lines = [f"Search `{query}`:"]
            for uid, sim in zip(uids, sims, strict=True):
                asset_id = str(uid)
                anno = ObjectMeta.annotation(asset_id) or {}
                lines.append(f"- {float(sim):.3f} `{asset_id}` {anno.get('category', '')}")
                if asset_id in self.objects:
                    self.object_query.value = query
                    self.load_object(asset_id)
                    self.status_md.content = "\n".join(lines[:8])
                    return
            self.set_status("\n".join(lines[:8]) + "\n\nNo hit has a local `thor` USD.")
        except Exception as exc:
            self.set_status(f"Object search failed: `{exc}`")

    def combined_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        if self.scene_mesh is not None and bool(self.show_scene.value):
            scene_min, scene_max = self.scene_mesh.bounds
            mins.append(scene_min)
            maxs.append(scene_max)
        if self.object_mesh is not None and bool(self.show_object.value):
            object_min, object_max = self.object_mesh.bounds
            mins.append(object_min)
            maxs.append(object_max)
        if not mins:
            return None
        return np.stack(mins).min(axis=0), np.stack(maxs).max(axis=0)

    def frame_client(self, client: viser.ClientHandle) -> None:
        bounds = self.combined_bounds()
        if bounds is None:
            return
        bounds_min, bounds_max = bounds
        center = 0.5 * (bounds_min + bounds_max)
        extent = float(np.linalg.norm(bounds_max - bounds_min))
        distance = max(1.5, 0.8 * extent)
        position = center + np.asarray([0.65 * distance, -0.85 * distance, 0.45 * distance], dtype=np.float32)
        try:
            with client.atomic():
                client.camera.position = position
                client.camera.look_at = center
                client.camera.up_direction = (0.0, 0.0, 1.0)
        except AssertionError:
            return

    def frame_all_clients(self) -> None:
        for client in self.server.get_clients().values():
            self.frame_client(client)

    def run(self) -> None:
        url_host = "localhost" if self.args.host in {"0.0.0.0", "::"} else self.args.host
        print(f"Viser MolmoSpaces viewer running at http://{url_host}:{self.args.port}")
        print("Use the Scene and Object controls in the browser to switch USDs.")
        while True:
            time.sleep(0.1)


def main() -> None:
    viewer = MolmoViserViewer(parse_args())
    viewer.run()


if __name__ == "__main__":
    main()
