import torch


def adjusted_constraint_loss(predictions: torch.Tensor,
                             ground_truth: torch.Tensor,
                             masks: dict[int, torch.Tensor | None],
                             reduction: str = 'mean'):

    if reduction not in ('sum', 'mean', 'none'):
        raise ValueError(f"Adjusted Constraint Loss Error: Invalid reduction type; {reduction}")
    device = predictions.device

    prediction_errors = predictions - ground_truth
    squared_errors = prediction_errors.pow(2)
    error_signs = torch.sign(prediction_errors)

    # Clone and adjust the terms in-place
    adjusted_loss_terms = squared_errors.clone()

    for var_idx, mask in masks.items():
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device)
            corrections = mask.nonzero()

            if corrections.numel() > 0:
                correction_indices = corrections[:, 0]
                mask_indices = corrections[:, 1]
                var_signs = error_signs[correction_indices, var_idx]
                mask_signs = error_signs[correction_indices, mask_indices]
                adjusted_loss_terms[correction_indices, var_idx] *= var_signs * mask_signs

    if reduction == "mean":
        return torch.mean(adjusted_loss_terms)
    elif reduction == "sum":
        return torch.sum(adjusted_loss_terms)

    return adjusted_loss_terms
