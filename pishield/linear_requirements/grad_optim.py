import torch
import random
from typing import List, Union, Tuple, Dict
from classes import Constraint, Variable

class GradOptimLayer(torch.nn.Module):
    def __init__(self, constraints: List[Constraint]):
        super().__init__()

        # Don't touch variables that don't have involvement of constraints

        self.constraints = constraints
        self.anchor_masks = {}

    def __call__(self, preds: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Optimizes gradients based on constraints and current predictions.

        This function works under the condition that all anchor variables are
        able to be sorted in their unique constraint sets, and do not appear
        in other constraint sets. This is an edge case which needs to be ironed
        out

        Args:
            preds (torch.Tensor): Current model predictions
            ground_truth (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Optimized Modified gradients
        """
        optimized_preds = preds.clone()

        # Group constraints by anchor variable
        constraint_groups = {}
        for constraint in self.constraints:
            # Always pick the first variable arbitrarily, after reduction it
            # is assumed that cyclical dependencies are broken?
            anchor = constraint.single_inequality.body[0].variable
            constraint_groups.setdefault(anchor, []).append(constraint)

        # Process each anchor variable
        for anchor, constraints in constraint_groups.items():
            anchor_id = anchor.id

            # At each iteration, improve the prediction and assign the mask
            optimized_preds, mask = self.__apply_mask(
                constraints,
                optimized_preds,
                ground_truth,
                anchor_id
            )
            self.anchor_masks[anchor_id] = mask

        return optimized_preds

    def __apply_mask(
        self,
        constraints: List[Constraint],
        preds: torch.Tensor,
        ground_truth: torch.Tensor,
        anchor_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies and returns a mask based on the constraints, for one anchor
        variable.

        Args:
            constraints (List[Constraint]): List of constraints
            preds (torch.Tensor): Current model predictions
            ground_truth (torch.Tensor): Ground truth labels
            anchor_id (int): The ID of the anchor variable to avoid masking

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Masked predictions and mask
        """
        final_mask = 0

        for constraint in constraints:
            inequality = constraint.single_inequality

            # Ensure that the anchor variable itself is not a candidate for the mask
            candidates = list(range(len(inequality.body)))
            candidates.remove(anchor_id)

            # Select random atom to mask (one-hot encoding)
            # TODO: Try other methods other than random
            mask_index = random.choice(candidates)
            mask = inequality.body[mask_index].variable.get_variable_id()

            # Correction is initialized with the mask term in the current prediction, which does not change
            correction = preds[:, mask].clone()

            # All other terms come from the ground truth values
            for atom in inequality.body:
                # y_1 serves as anchor, so no correction involvement
                if atom.variable.id == mask:
                    continue

                # Opposite signs, because the negative of the inequality (barring the anchor) is being assessed.
                # Original structure is y_1 - y_2 - y_3 > 0.
                sign_factor = -1 if atom.positive_sign else 1
                correction += ground_truth[:, atom.variable.id] * atom.coefficient * sign_factor

            # Update with maximum value based on the correction
            # TODO: Experiment with different forms of comparison? Max probably won't work here, unless element-wise
            original_norm = torch.norm(preds[:, anchor_id])
            corrected_norm = torch.norm(correction)

            # Overall magnitude comparison
            if corrected_norm > original_norm:
                preds[:, anchor_id] = correction
                # New mask needs to belong to the inequality that is used out of the options for the anchor variable
                final_mask = mask

        return preds, final_mask
