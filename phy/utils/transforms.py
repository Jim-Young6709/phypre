"""
--------------------------------------------
NOT THOROUGHLY REVIEWED Codex generated scripts.
--------------------------------------------

simulator-independent Torch transform helpers.
"""

from __future__ import annotations

import math

import torch

ASSET_AXIS_CONVENTIONS = ("y_up_to_z_up", "usd")


def _validate_tensor(
    value: torch.Tensor, trailing_shape: tuple[int, ...], name: str
) -> None:
    """Validate a floating tensor's trailing dimensions.

    Args:
        value: Tensor shaped ``(..., *trailing_shape)``.
        trailing_shape: Required trailing dimensions.
        name: Input name used in error messages.

    Returns:
        None.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if (
        value.ndim < len(trailing_shape)
        or value.shape[-len(trailing_shape) :] != trailing_shape
    ):
        raise ValueError(
            f"Expected {name} shape (..., {', '.join(map(str, trailing_shape))}), "
            f"got {tuple(value.shape)}."
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}.")


def _normalize_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize scalar-first quaternions.

    Args:
        quaternion: Tensor shaped ``(..., 4)`` in ``(w, x, y, z)`` order.

    Returns:
        Tensor shaped ``(..., 4)``. Zero quaternions become identity.
    """
    scale = quaternion.abs().amax(dim=-1, keepdim=True)
    scaled = quaternion / torch.where(scale == 0.0, torch.ones_like(scale), scale)
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    normalized = scaled / torch.where(norm == 0.0, torch.ones_like(norm), norm)
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1.0
    return torch.where(scale == 0.0, identity, normalized)


def _sqrt_positive_part(value: torch.Tensor) -> torch.Tensor:
    """Compute ``sqrt(max(value, 0))`` with a zero derivative at zero.

    Args:
        value: Tensor with any shape ``(...)``.

    Returns:
        Tensor with the same shape as ``value``.
    """
    result = torch.zeros_like(value)
    positive = value > 0.0
    result[positive] = torch.sqrt(value[positive])
    return result


def axis_rotation_wxyz(
    convention: str,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create the quaternion that corrects an asset's up axis.

    Args:
        convention: ``"y_up_to_z_up"`` or ``"usd"``.
        dtype: Output floating-point dtype.
        device: Output device.

    Returns:
        Quaternion tensor shaped ``(4,)`` in ``(w, x, y, z)`` order.
    """
    if convention == "y_up_to_z_up":
        values = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    elif convention == "usd":
        values = (1.0, 0.0, 0.0, 0.0)
    else:
        raise ValueError(
            f"Unknown asset axis convention {convention!r}; "
            f"expected one of {ASSET_AXIS_CONVENTIONS}."
        )

    quaternion = torch.tensor(values, dtype=dtype, device=device)
    if not quaternion.is_floating_point():
        raise TypeError(f"dtype must be floating point, got {quaternion.dtype}.")
    return quaternion


def wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first quaternions to rotation matrices.

    Args:
        quaternion: Tensor shaped ``(..., 4)`` in ``(w, x, y, z)`` order.

    Returns:
        Rotation matrices shaped ``(..., 3, 3)``.
    """
    _validate_tensor(quaternion, (4,), "quaternion")
    w, x, y, z = _normalize_wxyz(quaternion).unbind(dim=-1)

    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to scalar-first quaternions.

    Args:
        rotation: Tensor shaped ``(..., 3, 3)``.

    Returns:
        Normalized quaternions shaped ``(..., 4)`` in ``(w, x, y, z)`` order.
    """
    _validate_tensor(rotation, (3, 3), "rotation")
    batch_shape = rotation.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rotation.reshape(
        batch_shape + (9,)
    ).unbind(dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    candidates = torch.stack(
        (
            torch.stack(
                (q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01),
                dim=-1,
            ),
            torch.stack(
                (m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20),
                dim=-1,
            ),
            torch.stack(
                (m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21),
                dim=-1,
            ),
            torch.stack(
                (m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., :, None].clamp_min(0.1))
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(best.shape + (1, 4))
    return _normalize_wxyz(candidates.gather(-2, gather_index).squeeze(-2))


def transform_from_pos_wxyz(
    position: torch.Tensor, orientation: torch.Tensor
) -> torch.Tensor:
    """Build homogeneous transforms from positions and orientations.

    Args:
        position: Translation tensor shaped ``(..., 3)``.
        orientation: Quaternion tensor shaped ``(..., 4)`` in ``(w, x, y, z)``
            order. Its batch dimensions broadcast with ``position``.

    Returns:
        Homogeneous transforms shaped ``(..., 4, 4)`` with the broadcast batch
        dimensions.
    """
    _validate_tensor(position, (3,), "position")
    _validate_tensor(orientation, (4,), "orientation")
    if position.dtype != orientation.dtype or position.device != orientation.device:
        raise ValueError(
            "position and orientation must have the same dtype and device."
        )

    rotation = wxyz_to_matrix(orientation)
    batch_shape = torch.broadcast_shapes(position.shape[:-1], rotation.shape[:-2])
    upper = torch.cat(
        (
            rotation.expand(batch_shape + (3, 3)),
            position.expand(batch_shape + (3,)).unsqueeze(-1),
        ),
        dim=-1,
    )
    bottom = position.new_tensor((0.0, 0.0, 0.0, 1.0)).expand(batch_shape + (1, 4))
    return torch.cat((upper, bottom), dim=-2)


def poses_from_transforms(transforms: torch.Tensor) -> torch.Tensor:
    """Convert homogeneous transforms to position-quaternion poses.

    Args:
        transforms: Tensor shaped ``(..., 4, 4)``.

    Returns:
        Poses shaped ``(..., 7)`` in ``(x, y, z, qw, qx, qy, qz)`` order.
    """
    _validate_tensor(transforms, (4, 4), "transforms")
    return torch.cat(
        (
            transforms[..., :3, 3],
            matrix_to_wxyz(transforms[..., :3, :3]),
        ),
        dim=-1,
    )


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices.

    Args:
        axis_angle: Tensor shaped ``(..., 3)``. Direction is the rotation axis;
            magnitude is the angle in radians.

    Returns:
        Rotation matrices shaped ``(..., 3, 3)``.
    """
    _validate_tensor(axis_angle, (3,), "axis_angle")
    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        axis_angle.shape[:-1] + (3, 3)
    )
    angle = torch.linalg.vector_norm(axis_angle, dim=-1)
    sin_over_angle = torch.sinc(angle / math.pi)[..., None, None]
    one_minus_cos_over_angle_sq = (0.5 * torch.sinc(angle / (2.0 * math.pi)).square())[
        ..., None, None
    ]
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(
        skew
    )
    return (
        identity + sin_over_angle * skew + one_minus_cos_over_angle_sq * (skew @ skew)
    )


def offset_and_perturb_transforms(
    transforms: torch.Tensor,
    offset: float,
    axis: str,
    position_noise: float,
    rotation_noise: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Copy transforms and apply a local-axis offset and Gaussian pose noise.

    Args:
        transforms: Tensor shaped ``(..., 4, 4)``.
        offset: Distance along ``axis``.
        axis: Signed local axis: ``+x``, ``-x``, ``+y``, ``-y``, ``+z``, or
            ``-z``.
        position_noise: Standard deviation of world-frame translation noise.
        rotation_noise: Standard deviation of local-frame axis-angle noise.
        generator: Torch random generator matching the tensor device.

    Returns:
        Perturbed transforms shaped ``(..., 4, 4)``. The input is unchanged.
    """
    _validate_tensor(transforms, (4, 4), "transforms")
    if not isinstance(generator, torch.Generator):
        raise TypeError(
            f"generator must be a torch.Generator, got {type(generator).__name__}."
        )
    if axis not in ("+x", "-x", "+y", "-y", "+z", "-z"):
        raise ValueError(f"Unknown signed axis {axis!r}.")

    result = transforms.clone()
    rotation = transforms[..., :3, :3]
    axis_index = "xyz".index(axis[1])
    axis_sign = -1.0 if axis[0] == "-" else 1.0
    result[..., :3, 3] += axis_sign * offset * rotation[..., :, axis_index]

    batch_shape = transforms.shape[:-2]
    rotation_perturbation = None
    if position_noise > 0.0 and rotation_noise > 0.0:
        noise = torch.randn(
            batch_shape + (2, 3),
            dtype=transforms.dtype,
            device=transforms.device,
            generator=generator,
        )
        result[..., :3, 3] += noise[..., 0, :] * position_noise
        rotation_perturbation = noise[..., 1, :] * rotation_noise
    elif position_noise > 0.0:
        result[..., :3, 3] += (
            torch.randn(
                batch_shape + (3,),
                dtype=transforms.dtype,
                device=transforms.device,
                generator=generator,
            )
            * position_noise
        )
    elif rotation_noise > 0.0:
        rotation_perturbation = (
            torch.randn(
                batch_shape + (3,),
                dtype=transforms.dtype,
                device=transforms.device,
                generator=generator,
            )
            * rotation_noise
        )

    if rotation_perturbation is not None:
        result[..., :3, :3] = rotation @ _axis_angle_to_matrix(rotation_perturbation)
    return result
