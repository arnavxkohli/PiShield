import torch


def adjusted_constraint_loss(output: torch.Tensor,
                             target: torch.Tensor,
                             masks: dict[int, torch.Tensor | None],
                             reduction: str = 'mean'):
    """
    Calculates a modified MSE loss that works in tandem with the masking logic
    present in the shield layer. Where there is no adjustment made, the masks
    dictionary contains null values, and this simplifies to a regular MSE
    for that variable. In case a variable has been adjusted, the masked variable
    is picked out as a guide for the gradient's direction. This function ensures
    that gradients either stay the same or move in the same direction as they would
    without an augmented shield layer.

    Args:
        output: The model's output tensor.
        target: The target tensor.
        masks: A dictionary where keys are variable indices and values are tensors
               indicating which other variables' errors influence the loss adjustment
               for the key variable. A non-zero entry mask[i, j] means variable j's
               error sign affects variable i's loss term.
        reduction: Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.

    Returns:
        The calculated adjusted loss tensor.
    """
    # Ensure a valid reduction method is specified.
    if reduction not in ('sum', 'mean', 'none'):
        raise ValueError(f"Adjusted Constraint Loss Error: Invalid reduction type; {reduction}")
    device = output.device

    # Calculate basic error components needed for adjustment.
    prediction_errors = output - target
    squared_errors = prediction_errors.pow(2) # Base MSE calculation.
    error_signs = torch.sign(prediction_errors) # Needed to determine adjustment direction.

    # Start with standard squared errors; adjustments be applied in-place.
    adjusted_loss_terms = squared_errors.clone()

    # Iterate through each variable that might have its loss term adjusted.
    for var_idx, mask in masks.items():
        # Process only if a valid mask tensor exists for this variable.
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device)
            # Find samples (rows) and influencing variables (columns) requiring adjustment.
            corrections = mask.nonzero()

            # Apply adjustments only if there are corrections to be made for this variable.
            if corrections.numel() > 0:
                correction_indices = corrections[:, 0] # Sample indices needing adjustment.
                mask_indices = corrections[:, 1]       # Indices of variables influencing the adjustment.

                # Get the signs of errors for the variable being adjusted and the influencing variable.
                var_signs = error_signs[correction_indices, var_idx]
                mask_signs = error_signs[correction_indices, mask_indices]

                # Modify the loss term
                adjusted_loss_terms[correction_indices, var_idx] *= var_signs * mask_signs

    # Aggregate the adjusted loss terms based on the specified reduction method.
    if reduction == "mean":
        return torch.mean(adjusted_loss_terms)
    elif reduction == "sum":
        return torch.sum(adjusted_loss_terms)
    return adjusted_loss_terms
