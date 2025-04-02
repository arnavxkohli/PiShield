import torch
import torch.nn as nn
from typing import Dict, List
from pishield.linear_requirements.classes import Constraint

class LinearConstraintGradientLoss(nn.Module):
    """
    Computes a variant of MSE loss incorporating linear constraints
    represented by anchor-mask variable pairs.

    The loss term for an anchor variable is its squared error, multiplied
    by a sign factor derived from the agreement of error signs between
    the anchor and its corresponding masked variable.

    Loss Term (for anchor `a` and mask `m`):
        (pred_a - gt_a)^2 * sign(pred_a - gt_a) * sign(pred_m - gt_m)

    The total loss is the reduction (sum or mean) of these terms over
    all anchor-mask pairs defined in `anchor_mask` and across the batch.

    Attributes:
        anchor_mask (Dict[int, int]): Dictionary mapping anchor variable
            indices to their corresponding masked variable indices.
        ordering (List[int]): List of variable indices, potentially used for
            custom normalization (though standard reduction is preferred).
            Length might be used if reduction='mean' requires specific scaling.
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self,
                 anchor_mask: Dict[int, int],
                 ordering: List[int],
                 reduction: str = 'mean'):
        super().__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction type: {reduction}")

        self.anchor_mask: Dict[int, int] = anchor_mask
        self.ordering: List[int] = ordering
        self.reduction: str = reduction
        self.anchor_ids: List[int] = sorted(list(self.anchor_mask.keys()))

    def forward(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Compute the Linear Constraint Gradient loss.

        Args:
            predictions (torch.Tensor): Predicted tensor (batch_size, num_variables).
            ground_truth (torch.Tensor): Ground truth tensor (batch_size, num_variables).

        Returns:
            torch.Tensor: The calculated loss (scalar or tensor based on reduction).
        """
        if predictions.shape != ground_truth.shape:
             raise ValueError(f"Shape mismatch: predicted {predictions.shape} vs ground truth {ground_truth.shape}")

        batch_size = predictions.shape[0]
        if batch_size == 0:
             return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

        prediction_errors = predictions - ground_truth
        accumulated_loss= torch.zeros(batch_size,
                                             device=predictions.device,
                                             dtype=predictions.dtype)

        # Iterate through the precomputed list of anchor IDs
        for anchor_id in self.anchor_ids:
            mask_id = self.anchor_mask[anchor_id]

            anchor_error = prediction_errors[:, anchor_id]
            mask_error = prediction_errors[:, mask_id]

            sign_factor = torch.sign(anchor_error) * torch.sign(mask_error)
            accumulated_loss += torch.pow(anchor_error, 2) * sign_factor

        # Apply final reduction across the batch
        loss: torch.Tensor
        if self.reduction == 'sum':
            loss = torch.sum(accumulated_loss)
        elif self.reduction == 'mean':
            loss = torch.mean(accumulated_loss)
        elif self.reduction == 'none':
            loss = accumulated_loss
        else:
            loss = torch.mean(accumulated_loss)

        return loss
