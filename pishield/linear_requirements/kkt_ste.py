import torch
from torch.autograd import Function
from typing import List, Tuple

from pishield.linear_requirements.shield_layer import ShieldLayer
from pishield.linear_requirements.classes import Constraint
from pishield.linear_requirements.constants import EPSILON


def build_constraint_matrix(
    constraints: List[Constraint],
    num_vars: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble the global constraint matrix A and vector b such that
        A @ y <= b
    encodes every linear inequality from PiShield’s Constraint objects.

    Args:
        constraints: list of PiShield Constraint objects
        num_vars:    dimensionality of the prediction vector y

    Returns:
        A: Tensor of shape (num_constraints, num_vars)
        b: Tensor of shape (num_constraints,)
    """
    rows, rhs = [], []
    for constr in constraints:
        row = torch.zeros(num_vars, dtype=torch.float32)
        ineq = constr.single_inequality

        # coefficients part
        for atom in ineq.body:
            sign = 1.0 if atom.positive_sign else -1.0
            row[atom.variable.id] = atom.coefficient * sign

        # constant term and strictness adjustment
        constant = float(ineq.constant)
        if ineq.ineq_sign == ">":
            constant += EPSILON
        elif ineq.ineq_sign == "<":
            constant -= EPSILON

        # flip >= constraints into <=
        if ineq.ineq_sign in (">", ">="):
            row = -row
            constant = -constant

        rows.append(row)
        rhs.append(constant)

    A = torch.stack(rows, dim=0)
    b = torch.tensor(rhs, dtype=torch.float32)
    return A, b


class KKTShieldSTE(Function):
    """
    A custom autograd Function that preserves PiShield’s forward clamp
    (max/min) logic but uses a KKT‐projection Jacobian in the backward pass
    to unblock gradients and enforce tangent‐space updates.
    """

    @staticmethod
    def forward(
        ctx,
        predictions: torch.Tensor,
        A: torch.Tensor,
        b: torch.Tensor,
        shield_layer: ShieldLayer
    ) -> torch.Tensor:
        """
        Forward: apply PiShield’s ShieldLayer under no_grad to guarantee
        constraint satisfaction exactly as before.
        """
        with torch.no_grad():
            corrected = shield_layer(predictions)
        A, b = A.to(predictions.device), b.to(predictions.device)
        ctx.save_for_backward(corrected, A, b)
        return corrected

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor  # dL/dy_corrected
    ) -> Tuple[torch.Tensor, None, None, None]:
        corrected, A, b = ctx.saved_tensors  # A, b are already on device
        B, _ = corrected.shape
        device = corrected.device
        grad_input = torch.zeros_like(grad_output, device=device)  # dL/dy_predictions

        # Relative strength of the ridge regularization
        relative_ridge_strength = 1e-7

        for i in range(B):
            y_s = corrected[i]           # (n_vars,)
            g_sample = grad_output[i]    # (n_vars,)

            # Identify active constraints: A @ y_s approx b (for A@y_s <= b form)
            residuals_from_boundary = b - (A @ y_s)  # Should be >= 0. Active if close to 0.
            active_mask = residuals_from_boundary <= EPSILON
            
            num_active = int(active_mask.sum().item())

            if num_active == 0:
                grad_input[i] = g_sample
            else:
                G = A[active_mask]                  # (k, n_vars), k = num_active
                GGt = G @ G.T                       # (k, k)
                
                # Adaptive ridge value
                mean_diag_ggt = GGt.diag().mean().clamp(min=1e-9) if num_active > 0 else torch.tensor(1e-9, device=device, dtype=GGt.dtype)
                absolute_ridge_val = relative_ridge_strength * mean_diag_ggt
                reg_matrix = absolute_ridge_val * torch.eye(num_active, device=device, dtype=GGt.dtype)
                
                matrix_to_solve_on = GGt + reg_matrix
                target_for_solve = G @ g_sample.unsqueeze(1)  # (k, 1)

                try:
                    condition_num = torch.linalg.cond(matrix_to_solve_on)
                    if condition_num > 1e7:
                        solved_term = torch.linalg.lstsq(matrix_to_solve_on, target_for_solve, driver='gelsd').solution
                    else:
                        solved_term = torch.linalg.solve(matrix_to_solve_on, target_for_solve)  # (k, 1)
                except RuntimeError:
                    solved_term = torch.linalg.lstsq(matrix_to_solve_on, target_for_solve, driver='gelsd').solution  # Robust fallback
                
                grad_adjustment = G.T @ solved_term  # (n_vars, 1)
                grad_input[i] = g_sample - grad_adjustment.squeeze(1)  # (n_vars,)

        return grad_input, None, None, None
