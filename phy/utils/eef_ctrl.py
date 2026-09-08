import math

import torch
from isaaclab.utils import math as torch_utils


def compute_dof_pos_delta(arm_dof_pos: torch.Tensor,
                           current_eef_pos: torch.Tensor,
                           current_eef_quat: torch.Tensor,
                           jacobian: torch.Tensor,
                           ctrl_target_eef_pos: torch.Tensor,
                           ctrl_target_eef_quat: torch.Tensor,
                           ik_nullspace_target: torch.Tensor = None,
                           ik_nullspace_gain: float = 0.0,
                           damping: float = 0.1):
    """Compute Franka DOF position delta to move the end effector towards a target pose."""

    device = arm_dof_pos.device
    pos_error, axis_angle_error = get_pose_error(
        current_eef_pos=current_eef_pos,
        current_eef_quat=current_eef_quat,
        ctrl_target_eef_pos=ctrl_target_eef_pos,
        ctrl_target_eef_quat=ctrl_target_eef_quat)

    delta_eef_pose = torch.cat((pos_error, axis_angle_error), dim=1)
    delta_arm_dof_pos = _get_delta_dof_pos(delta_pose=delta_eef_pose,
                                           ik_method="dls",
                                           jacobian=jacobian,
                                           device=device,
                                           dof_pos=arm_dof_pos,
                                           ik_nullspace_target=ik_nullspace_target,
                                           ik_nullspace_gain=ik_nullspace_gain,
                                           damping=damping)

    return delta_arm_dof_pos


def get_pose_error(current_eef_pos,
                   current_eef_quat,
                   ctrl_target_eef_pos,
                   ctrl_target_eef_quat):
    """Compute task-space error between the target and current end-effector poses."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf

    # Compute pos error
    pos_error = ctrl_target_eef_pos - current_eef_pos

    # Compute rot error
    # Compute quat error (i.e., difference quat)
    # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html
    current_eef_quat_norm = torch_utils.quat_mul(current_eef_quat,
                                                 torch_utils.quat_conjugate(current_eef_quat))[:, 0]  # scalar component
    current_eef_quat_inv = torch_utils.quat_conjugate(
        current_eef_quat) / current_eef_quat_norm.unsqueeze(-1)
    quat_error = torch_utils.quat_mul(ctrl_target_eef_quat, current_eef_quat_inv)

    # Convert to axis-angle error
    axis_angle_error = axis_angle_from_quat(quat_error)

    return pos_error, axis_angle_error


def _get_delta_dof_pos(delta_pose,
                       ik_method,
                       jacobian,
                       device,
                       k_val=1.0,
                       dof_pos=None,
                       ik_nullspace_target=None,
                       ik_nullspace_gain=0.0,
                       damping=0.1):
    """Get delta Franka DOF position from delta pose using specified IK method."""
    # References:
    # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

    if ik_method == 'pinv':  # Jacobian pseudoinverse
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == 'trans':  # Jacobian transpose
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == 'dls':  # damped least squares (Levenberg-Marquardt)
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (damping ** 2) * torch.eye(n=jacobian.shape[1], device=device)
        jacobian_pinv = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
        delta_dof_pos = jacobian_pinv @ delta_pose.unsqueeze(-1)

        if ik_nullspace_target is not None and ik_nullspace_gain > 0.0:
            num_dofs = jacobian.shape[2]
            identity = torch.eye(n=num_dofs, device=device).unsqueeze(0)
            nullspace_error = (ik_nullspace_target - dof_pos).unsqueeze(-1)
            delta_dof_pos = delta_dof_pos + ik_nullspace_gain * (identity - jacobian_pinv @ jacobian) @ nullspace_error

        delta_dof_pos = k_val * delta_dof_pos.squeeze(-1)

    elif ik_method == 'svd':  # adaptive SVD
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1. / S
        min_singular_value = 1.0e-5
        S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos


def axis_angle_from_quat(quat, eps=1.0e-6):
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference: https://github.com/facebookresearch/pytorch3d/blob/bee31c48d3d36a8ea268f9835663c52ff4a476ec/pytorch3d/transforms/rotation_conversions.py#L516-L544

    quat = quat * (1.0 - 2.0 * (quat[:, 0:1] < 0.0))
    mag = torch.linalg.norm(quat[:, 1:4], dim=1)
    half_angle = torch.atan2(mag, quat[:, 0])
    angle = 2.0 * half_angle
    sin_half_angle_over_angle = torch.where(torch.abs(angle) > eps,
                                            torch.sin(half_angle) / angle,
                                            1 / 2 - angle ** 2.0 / 48)
    axis_angle = quat[:, 1:4] / sin_half_angle_over_angle.unsqueeze(-1)

    return axis_angle
