import torch
import torch.nn as nn
from pishield.linear_requirements.constants import EPSILON, INFINITY

'''
- TODO: Fix the implementation for anchor coefficients which are not 1. Could be
  -ve, in which case use min.
- Ensure the dimensions work well. Need to squeeze back to 2D if moving to 3D.
'''

class AdjustedConstraintLoss(nn.Module):
    """
    Computes an adjusted element-wise loss, guided by anchor masks from constraint correction.

    This loss module wraps the logic of `adjusted_constraint_loss` into a standard
    PyTorch `nn.Module` interface. It modifies a base MSE loss based on constraint
    corrections applied in a prior layer (`GradOptimLayer`).

    The loss for an element corresponding to anchor variable A is calculated as:

        L[b, n, d] = MSE(h_A'[b,n,d], y_A[b,n,d]) *
                     sign(h_M[b,m,d] - y_M[b,m,d]) *  (if corrected, m = anchor_masks[b,n,d])
                     sign(h_A'[b,n,d] - y_A[b,n,d])

    If no correction was applied (`anchor_masks[b,n,d] == -1`), the multiplier signs
    effectively become 1, and the loss reduces to standard MSE for that element.

    Attributes:
        reduction (str): Specifies the reduction to apply to the output:
            'none', 'mean', 'sum'. Default: 'mean'.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                'none': No reduction. Returns the full loss tensor.
                'mean': The sum of the output will be divided by the number of elements.
                'sum': The output will be summed. Default: 'mean'.
        """
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"AdjustedConstraintLoss: Invalid reduction type: {reduction}")
        self.reduction = reduction

    def forward(self,
                predictions: torch.Tensor,
                ground_truth: torch.Tensor,
                anchor_masks: torch.Tensor) -> torch.Tensor:
        """
        Computes the adjusted loss.

        Args:
            predictions (torch.Tensor): Corrected model predictions (h').
                Shape: (batch_size, num_variables[, variable_size])
            ground_truth (torch.Tensor): Ground truth labels (y).
                Shape: Same as `predictions`
            anchor_masks (torch.Tensor): Element-wise tensor containing mask variable IDs (m)
                used to compute the constraint correction for each anchor prediction (A).
                - Shape: Same as `predictions`
                - Each element is either:
                    - An integer mask variable ID (m)
                    - -1 if no correction was applied for element [b,n,d]

        Returns:
            torch.Tensor: The computed adjusted loss.
        """
        # Ensure all input tensors have matching shapes
        for name, tensor in [('ground_truth', ground_truth), ('anchor_masks', anchor_masks)]:
            if tensor.shape != predictions.shape:
                raise ValueError(
                    f"Adjusted Constraint Loss: shape mismatch: predictions {predictions.shape} "
                    f"vs {name} {tensor.shape}"
                )

        is_2d_input = False

        # If 2D, promote to 3D
        if predictions.ndim == 2:
            predictions = predictions.unsqueeze(-1)
            ground_truth = ground_truth.unsqueeze(-1)
            anchor_masks = anchor_masks.unsqueeze(-1)
            is_2d_input = True
        elif predictions.ndim != 3:
            raise ValueError(
                f"Adjusted Constraint Loss: Expected 2D or 3D tensors, got shape {predictions.shape}"
            )

        # Setup error signs and mse
        prediction_errors = predictions - ground_truth
        squared_errors = prediction_errors.pow(2)
        error_signs = torch.sign(prediction_errors)

        # Identify elements where constraint correction was applied
        has_constraint_correction = anchor_masks > -1

        # For anchor variables, we always use their own error sign
        anchor_error_signs = torch.where(has_constraint_correction,
                                         error_signs,
                                         torch.ones_like(error_signs))

        # For elements with constraint correction, we need to get the sign from the mask variable
        # Convert -1 indices to 0 to avoid out-of-bounds errors during gather
        safe_mask_indices = torch.where(has_constraint_correction,
                                        anchor_masks,
                                        torch.zeros_like(anchor_masks))

        # Gather error signs for the mask variables
        mask_error_signs = torch.gather(error_signs, dim=1, index=safe_mask_indices)

        # Fallback to ensure signs without correction are always +ve
        mask_error_signs[~has_constraint_correction] = 1
        anchor_error_signs[~has_constraint_correction] = 1

        adjusted_loss = squared_errors * anchor_error_signs * mask_error_signs

        # If originally 2D, convert back
        if is_2d_input:
            adjusted_loss = adjusted_loss.squeeze(-1)

        # Apply reduction if needed
        if self.reduction == 'mean':
            return adjusted_loss.mean()
        elif self.reduction == 'sum':
            return adjusted_loss.sum()
        return adjusted_loss
