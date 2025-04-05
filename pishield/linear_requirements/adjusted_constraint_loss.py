import torch

'''
- TODO: Fix the implementation for anchor coefficients which are not 1. Could be
  -ve, in which case use min.
- Ensure the dimensions work well. Need to squeeze back to 2D if moving to 3D.
'''

def adjusted_constraint_loss(predictions: torch.Tensor,
                             ground_truth: torch.Tensor,
                             anchor_masks: torch.Tensor,
                             reduction: str = 'mean') -> torch.Tensor:
    """
    Computes an adjusted element-wise loss for predictions, guided by anchor masks
    generated from constraint correction logic via `GradOptimLayer`.

    The loss formula is defined as:

        L[b, n, d] = MSE(h_A'[b,n,d], y_A[b,n,d]) *
                     sign(h_M[b,m,d] - y_M[b,m,d]) *
                     sign(h_A'[b,n,d] - y_A[b,n,d])

    where:
        - h_A' is the corrected prediction for anchor variable A
        - h_M is the prediction of the masked variable (used in the constraint correction)
        - y_A, y_M are ground truth values
        - The signs are used to adjust gradient flow based on the direction of constraint violations
        - If no correction was applied (`anchor_masks[b,n,d] == -1`), the loss reduces to standard MSE

    Args:
        predictions (torch.Tensor): Corrected model predictions (h').
            Shape: (batch_size, num_variables[, variable_size])
        ground_truth (torch.Tensor): Ground truth labels (y).
            Shape: Same as `predictions`
        anchor_masks (torch.Tensor): Element-wise tensor containing mask variable IDs (mask_id),
            used to compute the constraint correction for each anchor prediction.
            - Shape: Same as `predictions`
            - Each element is either:
                - An integer mask variable ID (used for correction)
                - -1 if no correction was applied
        reduction (str): Specifies the reduction to apply to the output:
            - 'none': No reduction. Return the full tensor.
            - 'mean': Return the mean of all losses.
            - 'sum': Return the sum of all losses.

    Returns:
        torch.Tensor: The computed adjusted loss.
            - Scalar if reduction is 'mean' or 'sum'.
            - Tensor of shape (batch_size, num_variables[, variable_size]) if reduction is 'none'.

    Raises:
        ValueError: If predictions and ground_truth have mismatched shapes or
                    if an invalid reduction type is specified.
    """
    if reduction not in ['none', 'mean', 'sum']:
        raise ValueError(f"Adjusted Constraint Loss: Invalid reduction type: {reduction}")

    # Ensure all input tensors have matching shapes
    for name, tensor in [('ground_truth', ground_truth), ('anchor_masks', anchor_masks)]:
        if tensor.shape != predictions.shape:
            raise ValueError(
                f"Adjusted Constraint Loss: shape mismatch: predictions {predictions.shape} "
                f"vs {name} {tensor.shape}"
            )

    dimension_change = False

    # If 2D, promote to 3D by adding a trailing dimension
    if predictions.ndim == 2:
        predictions = predictions.unsqueeze(-1)
        ground_truth = ground_truth.unsqueeze(-1)
        anchor_masks = anchor_masks.unsqueeze(-1)
        dimension_change = True
    elif predictions.ndim != 3:
        raise ValueError(
            f"Adjusted Constraint Loss: Expected 2D or 3D tensors, got shape {predictions.shape}"
        )

    # Compute standard per-element squared error
    prediction_deviations = predictions - ground_truth
    mse = prediction_deviations.pow(2)
    all_signs = torch.sign(prediction_deviations)

    # Element-wise mask indicator: where a correction was applied
    is_masked = anchor_masks > -1

    # Sign of anchor deviation is always used
    anchor_signs = torch.where(is_masked, all_signs, torch.ones_like(all_signs))

    # Gather from 0 index if -1. If greater, gather from the correct mask index
    mask_indices = torch.where(is_masked, anchor_masks, torch.zeros_like(anchor_masks))
    mask_signs = torch.gather(all_signs, dim=1, index=mask_indices)

    # Ensure +ve sign for non-masked elements
    mask_signs[~is_masked] = 1

    # MSE_A * sign(h_A' - y_A) * sign(h_M - y_M)
    adjusted_loss = mse * anchor_signs * mask_signs

    # Squeeze back to 2D if input was originally 2D
    if dimension_change:
        adjusted_loss = adjusted_loss.squeeze(-1)

    # Apply reduction if needed
    if reduction == 'mean':
        return adjusted_loss.mean()
    elif reduction == 'sum':
        return adjusted_loss.sum()
    return adjusted_loss
