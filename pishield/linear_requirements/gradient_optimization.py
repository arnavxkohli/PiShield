import torch
import random
from typing import List, Tuple, Dict
from pishield.linear_requirements.classes import Constraint, Variable
from pishield.linear_requirements.constants import EPSILON, INFINITY

'''
- TODO: Fix the implementation for anchor coefficients which are not 1. Could be
  -ve, in which case use min.
- Need to figure out sequential ordering of structured constraints like y_1 > y_0 and y_2 > y_1
'''

class GradOptimLayer(torch.nn.Module):
    """
    A PyTorch layer that adjusts model predictions element-wise based on
    linear inequality constraints. This layer currently works for two and
    three-dimensional tensors.

    This layer modifies predictions (h) based on constraints, aiming to satisfy them
    by tweaking the value of certain 'anchor' variables. The adjustment uses a mix of
    other predictions (h_M) and ground truth (y_NM) values. It calculates potential
    correction values derived from each relevant constraint and applies the
    maximum correction element-wise:
        h_A' = max(h_A, correction(constraint_1), correction(constraint_2), ...)

    The correction derived from a constraint uses one variable's prediction (masked variable h_M)
    and other variables' ground truth values (non-masked y_NM).

    A mask tensor is generated for the entire prediction tensor, where each element indicates
    which variable's prediction (mask_id) was used in the calculation that resulted in the final
    corrected value. A value of -1 indicates the original prediction h_A was kept
    (i.e., no correction applied for that element).

    Note:
        A key flaw with this layer is that it is unable to handle constraints of the construction:
        y_1 > y_0; y_2 > y_1.
        Ideally, this would be structued within the layer as:
            y_0' = y_0
            y_1' = max(y_1, y_0')
            y_2' = max(y_2, y_1')
        given the ordering y_0, y_1, y_2
        The layer uses the heuristic that the anchor variable is the first one in the body,
        and does not appear as a dependent variable in any other inequalities. An extension
        to this layer would be the case where the sequential ordering is respected.
        The current construction performed by this layer is:
            y_0' = y_0
            y_1' = max(y_1, y_0')
            y_2' = max(y_2, y_1)
        Note the y_1' vs y_1 difference visible in the two constructions.

    Attributes:
        constraint_groups (Dict[Variable, List[Constraint]]): Constraints grouped by
            their anchor variable (arbitrarily the first variable in the inequality body).
        anchor_masks (torch.Tensor): Tensor of shape (batch_size, num_variables, variable_size),
            where each element stores the ID of the variable (mask_id) used for correction, or -1
            if the value was left unchanged.
    """

    def __init__(self, constraints: List[Constraint]):
        """
        Initializes the GradOptimLayer.

        Args:
            constraints (List[Constraint]): The list of linear inequality constraints.
                Each constraint is expected to be represented in a form like
                sum(coefficient * variable) > 0 or similar, accessible via
                constraint.single_inequality.
        """
        super().__init__()
        if not constraints:
            raise ValueError("GradOptimLayer Error: No constraints provided.")

        self.constraint_groups: Dict[Variable, List[Constraint]] = {}
        for constraint in constraints:
            # Assumption: Anchor variable is the first one listed in the inequality body.
            anchor_var = constraint.single_inequality.body[0].variable
            self.constraint_groups.setdefault(anchor_var, []).append(constraint)

        self.anchor_masks = None

    def forward(self, preds: torch.Tensor, ground_truth: torch.Tensor | None = None) -> torch.Tensor:
        """
        Applies element-wise constraint-based adjustments to predictions.

        Args:
            preds (torch.Tensor): Current model predictions (h).
                Shape: (batch_size, num_variables, variable_size) or (batch_size, num_variables)
            ground_truth (torch.Tensor): Ground truth labels (y).
                Shape: (batch_size, num_variables, variable_size) or (batch_size, num_variables)

        Returns:
            torch.Tensor: Predictions adjusted element-wise based on constraints.
                Shape: (batch_size, num_variables, variable_size)

        Side Effects:
            Updates `self.anchor_masks` (torch.Tensor):
                - Shape: (batch_size, num_variables, variable_size)
                - Each element is:
                    - The mask variable ID (mask_id) used in the correction, if a correction was applied.
                    - -1 if no correction was applied for that element.
        """

        # If in training mode, or if the ground truth has not been passed,
        # return the predicitions as is.
        if not self.training or ground_truth is None:
            return preds

        if preds.shape != ground_truth.shape:
            raise ValueError(
                f"GradOptimLayer Error: predictions shape {preds.shape} does not match "
                f"ground truth shape {ground_truth.shape}"
            )

        dimension_change = False

        # If 2D, promote to 3D by adding a trailing dimension
        if preds.ndim == 2:
            preds = preds.unsqueeze(-1)
            ground_truth = ground_truth.unsqueeze(-1)
            dimension_change = True
        elif preds.ndim != 3:
            raise ValueError(
                f"GradOptimLayer Error: Expected 2D or 3D tensor for predictions, "
                f"got shape {preds.shape}"
            )

        optimized_preds = preds.clone()
        # Initialize anchor_masks with -1 (i.e., no correction)
        self.anchor_masks = torch.full_like(preds, -1, dtype=torch.long)

        # Apply corrections for each anchor variable group
        for anchor_var, anchor_constraints in self.constraint_groups.items():
            anchor_id = anchor_var.id

            corrected_anchor_preds, corrected_anchor_masks = self._apply_correction(
                constraints=anchor_constraints,
                preds=preds,
                ground_truth=ground_truth,
                anchor_id=anchor_id,
            )

            # Update predictions and masks
            optimized_preds[:, anchor_id] = corrected_anchor_preds
            self.anchor_masks[:, anchor_id] = corrected_anchor_masks

        # Squeeze back if the input was originally 2D
        if dimension_change:
            optimized_preds = optimized_preds.squeeze(-1)
            self.anchor_masks = self.anchor_masks.squeeze(-1)

        return optimized_preds

    def _apply_correction(
        self,
        constraints: List[Constraint],
        preds: torch.Tensor,
        ground_truth: torch.Tensor,
        anchor_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates corrections for a single anchor variable based on its constraints.

        Implements h_A' = max(h_A, correction(c1), correction(c2), ...) element-wise.

        Args:
            constraints: List of constraints involving the anchor variable.
            preds: Original model predictions (h).
                Shape: (batch_size, num_variables, variable_size)
            ground_truth: Ground truth labels (y).
                Shape: (batch_size, num_variables, variable_size)
            anchor_id: The ID of the anchor variable (A).

        Returns:
            A tuple containing:
            - corrected_anchor_preds (torch.Tensor): The updated predictions for the anchor variable.
              Shape: (batch_size, variable_size)
            - best_mask_ids (torch.Tensor): Tensor of shape (batch_size, variable_size)
              containing the ID of the variable (mask_id) used for the correction calculation
              that yielded the maximum value for each element, or -1 if no correction was applied.
        """
        anchor_predictions = preds[:, anchor_id].clone()

        # Initialize corrections and masks with default (unchanged) values for each
        all_corrections = [anchor_predictions]
        all_mask_ids = [torch.full_like(anchor_predictions, -1, dtype=torch.long)]

        for inequality in map((lambda c: c.single_inequality), constraints):
            # Possible for y_0 > 0 kind of construction.
            if len(inequality.body) <= 1:
                continue

            # TODO: Explore strategies other than random selection (e.g., based on violation magnitude?)
            mask_var_id = random.choice([
                atom for atom in inequality.body
                if atom.variable.id != anchor_id
            ]).variable.id

            # Initialize correction with original value for masked variable (h_M)
            correction = preds[:, mask_var_id].clone()

            # Strict vs non-strict inequality correction
            epsilon = torch.full_like(correction, EPSILON)
            if inequality.ineq_sign == '<':
                correction -= epsilon
            elif inequality.ineq_sign == '>':
                correction += epsilon
            else:
                epsilon = torch.zeros_like(correction, dtype=correction.dtype)

            for atom in inequality.body:
                atom_id = atom.variable.id
                # y_{non-masked} only
                if atom_id in [anchor_id, mask_var_id]:
                    continue

                # Opposite sign: y_A - y_B - y_C > 0 becomes y_A > y_B + y_C,
                # Working with the right side of the inequality always
                sign_factor = -1.0 if atom.positive_sign else 1.0
                correction += ground_truth[:, atom_id] * atom.coefficient * sign_factor # Not sure if this is correct

            all_corrections.append(correction)
            all_mask_ids.append(torch.full_like(correction, mask_var_id))

        # Vectorize corrections and masks for element-wise max and mask operations
        all_corrections = torch.stack(all_corrections)  # Shape: (num_constraints + 1, batch_size, variable_size)
        all_mask_ids = torch.stack(all_mask_ids)        # Same shape

        corrected_anchor_preds, mask_indices = torch.max(all_corrections, dim=0)  # Shape: (batch_size, variable_size)
        best_mask_ids = torch.gather(all_mask_ids, 0, mask_indices.unsqueeze(0)).squeeze(0)

        return corrected_anchor_preds, best_mask_ids
