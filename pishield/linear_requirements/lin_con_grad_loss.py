import torch
import torch.nn as nn
from typing import List, Dict
from classes import Constraint, Variable  # Assuming these are defined elsewhere

class LinConGradLoss(nn.Module):
    def __init__(self,
                 constraints: List[Constraint],
                 ordering: List[Variable],
                 anchor_mask: Dict[int, int],
                 reduction: str = 'sum'):
        super().__init__()

        if reduction not in ['sum', 'mean']:
            raise ValueError('PyTorch Error: Invalid Reduction type')

        self.constraints = constraints
        self.ordering = ordering
        self.anchor_mask = anchor_mask
        self.reduction = reduction

    @torch.jit.script
    def forward(self, predictions: torch.Tensor, ground_truth: torch.Tensor):
        """
        Compute the Linear Constraint Gradient loss between predicted and
        ground truth tensors.

        Args:
            predicted (torch.Tensor, optional): Predicted tensor (defaults to
            self.predicted_variables).
            ground_truth (torch.Tensor, optional): Ground truth tensor
            (defaults to self.ground_truth).

        Returns:
            torch.Tensor: The MSE loss (scalar) based on the reduction type.
        """
        if predictions.shape != ground_truth.shape:
            raise ValueError(f"Shape mismatch: predicted {predictions.shape} vs ground truth {ground_truth.shape}")

        prediction_errors = predictions - ground_truth

        for anchor_id, mask in self.anchor_mask.items():
            # sign = sgn(h_A' - y_A) x sgn(h_B - y_B) operation
            sign_factor = torch.mul(
                            torch.sign(prediction_errors[:, mask]),
                            torch.sign(prediction_errors[:, anchor_id]),
                        )

            squared_error = torch.pow(prediction_errors[:, anchor_id], 2)

            # Should be okay to modify this, but there are conditions around
            # the anchor variable. Have a look at the notion document.
            predictions[:, anchor_id] = torch.mul(squared_error, sign_factor)

        torch.nn.functional.mse_loss(predictions, reduction=self.reduction)
