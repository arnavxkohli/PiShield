import torch
import random
from typing import List, Tuple, Dict
from pishield.linear_requirements.classes import Constraint, Variable

class GradOptimLayer(torch.nn.Module):
    """
    A PyTorch layer that attempts to adjust model predictions based on
    pre-defined linear inequality constraints.

    This layer modifies predictions directly, which influences gradients during
    backpropagation if used within a model's forward pass before the loss
    calculation.

    Attributes:
        constraints (List[Constraint]): A list of constraint objects defining
                                         relationships between variables.
        anchor_masks (Dict[int, int]): Stores the ID of the variable
                                      that was randomly chosen to be 'masked'
                                      (used in correction) for each anchor,
                                      updated during the forward pass.
                                      Value is None if no update occurred for
                                      that anchor.
    """

    def __init__(self, constraints: List[Constraint]):
        """
        Initializes the GradOptimLayer.

        Args:
            constraints (List[Constraint]): The list of constraints to enforce.
        """
        super().__init__()
        if not constraints:
            raise ValueError("GradOptimLayer Error: No constraints provided.")
        self.constraints = constraints

        # Stores mask for each anchor variable (to be computed)
        self.anchor_masks: Dict[int, int] = {}

    def forward(self, preds: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Applies constraint-based adjustments to the predictions.

        Note:
            This implementation makes several simplifying assumptions:
            1. Anchor Selection: Arbitrarily chooses the first variable
               in each constraint's inequality body as the 'anchor'.
            2. Anchor Uniqueness Assumption: Assumes anchors identified this way
               can be uniquely grouped. If a variable appears first in multiple
               constraints intended for different conceptual anchors, the grouping
               might be incorrect. A more robust anchor identification strategy
               may be needed.
            3. Mask Selection: Randomly selects one non-anchor variable within
               a constraint to base the correction on.
            4. Update Condition: Uses a potentially heuristic comparison of
               vector norms (`torch.norm`) to decide whether to apply the
               correction. This compares overall magnitudes, not direct
               constraint satisfaction.

        Extension: To ensure that anchor variables do not appear in other constraints
        as dependent variable, a mapping from variables to constraints is needed.
        Anchors could then be efficiently chosen based on this (maybe with topo-sort
        or other graph-based methods).

        Args:
            preds (torch.Tensor): Current model predictions (batch_size, num_variables).
            ground_truth (torch.Tensor): Ground truth labels (batch_size, num_variables).

        Returns:
            torch.Tensor: Predictions potentially adjusted based on constraints.
        """
        optimized_preds = preds.clone()
        self.anchor_masks.clear() # Clear masks from previous forward passes

        # Group constraints by anchor variable
        constraint_groups: Dict[Variable, List[Constraint]] = {}

        # TODO: Implement checks to ensure anchor variables do not appear as
        # dependent variables in other constraints.
        # For now, assume this is true, and the first variable in the constraint
        # is always valid (anchor).

        for constraint in self.constraints:
            # Get the first variable in the constraint as the anchor
            anchor_var = constraint.single_inequality.body[0].variable

            # Group constraints by anchor variable
            constraint_groups.setdefault(anchor_var, []).append(constraint)


        for anchor_var, constraints_for_anchor in constraint_groups.items():
            anchor_id = anchor_var.id

            # Apply mask/correction logic for this anchor based on its constraints
            optimized_preds, mask_id = self._apply_correction(
                constraints_for_anchor,
                optimized_preds,
                ground_truth,
                anchor_id,
            )
            # Store the ID of the variable used for the final update
            self.anchor_masks[anchor_id] = mask_id

        return optimized_preds

    def _apply_correction(
        self,
        constraints: List[Constraint],
        preds: torch.Tensor,
        ground_truth: torch.Tensor,
        anchor_id: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Applies corrections for a single anchor variable based on its constraints.

        Iterates through constraints associated with the anchor. For each,
        it calculates a potential correction for the anchor's prediction. If
        the norm of the correction is greater than the norm of the current
        prediction for the anchor, the prediction is updated.

        Args:
            constraints (List[Constraint]): List of constraints involving the anchor.
            preds (torch.Tensor): Current model predictions (potentially updated
                                   by previous anchors).
            ground_truth (torch.Tensor): Ground truth labels.
            anchor_id (int): The ID of the anchor variable.

        Returns:
            Tuple[torch.Tensor, int]:
                - The potentially updated predictions tensor.
                - The ID of the variable used for masking in the *last*
                  successful update for this anchor, or None if no update occurred.
        """
        final_mask_id: int = 0

        for constraint in constraints:
            inequality = constraint.single_inequality

            if len(inequality.body) <= 1:
                continue

            # Identify potential variables to use for correction, exclude the
            # anchor variable itself
            candidate_indices = [
                idx for idx, atom in enumerate(inequality.body)
                if atom.variable.id != anchor_id
            ]

            # TODO: Explore strategies other than random selection (based
            # on violation magnitude?)
            mask_atom_index = random.choice(candidate_indices)
            mask_id = inequality.body[mask_atom_index].variable.get_variable_id()

            # Initialize correction with the current prediction of the masked variable
            correction = preds[:, mask_id].clone()

            # Add contributions from other (non-anchor, non-mask) variables using ground truth
            for atom_idx, atom in enumerate(inequality.body):
                var_id = atom.variable.id
                if var_id == anchor_id or var_id == mask_id:
                    continue

                # Opposite sign: y_A - y_B - y_C > 0 becomes y_A > y_B + y_C,
                # Working with the right side of the inequality always
                sign_factor = -1.0 if atom.positive_sign else 1.0
                correction += (
                    ground_truth[:, var_id] * atom.coefficient * sign_factor
                )

            # TODO: This norm comparison is heuristic. Explore alternatives.
            # Does element-wise comparison make more sense? torch.where?
            # Does comparing norm ensure constraint satisfaction? Unlikely directly.
            original_anchor_pred = preds[:, anchor_id] # Shape: (batch_size,)

            # Ensure dimensions match for norm calculation if needed, though norms produce scalars
            # Calculate norms across the batch dimension? Or element-wise?
            original_norm = torch.norm(original_anchor_pred)
            corrected_norm = torch.norm(correction)

            # If the norm of the calculated correction is larger, update the prediction
            if corrected_norm > original_norm:
                preds = preds.clone()
                preds[:, anchor_id] = correction
                final_mask_id = mask_id

        return preds, final_mask_id

    def __call__(self, preds: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        return self.forward(preds, ground_truth)
