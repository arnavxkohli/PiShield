import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from pishield.linear_requirements.classes import Constraint, Variable, Atom
from pishield.linear_requirements.constants import EPSILON, INFINITY

# --- Attention Helper Module (Simpler for 2D) ---
# (AttentionScoreNet2D remains the same as before)
class AttentionScoreNet2D(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, anchor_feat: torch.Tensor, mask_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((anchor_feat.unsqueeze(-1), mask_feat.unsqueeze(-1)), dim=-1)
        return self.net(combined)

class GradOptimLayerAttn2D(nn.Module):
    """
    GradOptimLayer variant for 2D inputs, using attention to select/weight mask
    variables (h_M) and ground_truth (y_NM) for other variables. Does not compute
    or store anchor masks.

    Calculates corrections for anchor A based on constraint k using an
    attention-weighted h_M and ground truth y_NM.
        h_A' = max(h_A, attn_correction(constraint_1), attn_correction(constraint_2), ...)
    """
    def __init__(self, constraints: List[Constraint], attention_hidden_dim: int = 16):
        super().__init__() # Ensure super().__init__() is called first
        if not constraints:
            raise ValueError("GradOptimLayerAttn2D Error: No constraints provided.")

        self.constraints = constraints
        self.constraint_groups: Dict[int, List[Constraint]] = {}
        self.anchor_ids = set()
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_nets = nn.ModuleDict()

        for i, constraint in enumerate(self.constraints):
            # Ensure this assignment happens *after* super().__init__()
            self.attention_nets[f'attn_score_{i}'] = AttentionScoreNet2D(self.attention_hidden_dim)
            anchor_var = None
            for atom in constraint.single_inequality.body:
                if atom.coefficient > 0 and atom.positive_sign:
                    anchor_var = atom.variable
                    break
            if anchor_var is None:
                anchor_var = constraint.single_inequality.body[0].variable
            self.anchor_ids.add(anchor_var.id)
            self.constraint_groups.setdefault(anchor_var.id, []).append(constraint)

        # Removed self.anchor_masks initialization

    def forward(self, preds: torch.Tensor, ground_truth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies element-wise constraint-based adjustments using attention and ground truth.

        Args:
            preds (torch.Tensor): Current model predictions (h). Shape: (batch_size, num_variables)
            ground_truth (Optional[torch.Tensor]): Ground truth labels (y). Required if self.training.
                                                  Shape: (batch_size, num_variables)

        Returns:
            torch.Tensor: Predictions adjusted element-wise. Shape: (batch_size, num_variables)
        """
        if preds.ndim != 2:
            raise ValueError(f"Expected 2D tensor (batch, num_vars), got shape {preds.shape}")

        if not self.training or ground_truth is None:
            return preds

        if ground_truth.shape != preds.shape:
            raise ValueError(f"Predictions shape {preds.shape} must match ground_truth shape {ground_truth.shape}")

        device = preds.device

        optimized_preds = preds.clone()
        # Removed anchor_masks initialization

        for anchor_id in self.anchor_ids:
            anchor_constraints = self.constraint_groups.get(anchor_id, [])
            if not anchor_constraints:
                continue

            for key in self.attention_nets:
                self.attention_nets[key].to(device)

            # Corrected call: _apply_correction_attn now only returns predictions
            corrected_anchor_preds = self._apply_correction_attn(
                anchor_id=anchor_id,
                constraints=anchor_constraints,
                preds=preds,
                ground_truth=ground_truth,
            )

            optimized_preds[:, anchor_id] = corrected_anchor_preds
            # Removed assignment to self.anchor_masks

        return optimized_preds

    def _apply_correction_attn(
        self,
        anchor_id: int,
        constraints: List[Constraint],
        preds: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor: # Changed return type
        """
        Calculates attention-weighted corrections using h_M and y_NM.
        Returns only the corrected predictions for the anchor variable.
        """
        anchor_predictions = preds[:, anchor_id] # Shape: (batch,)

        all_corrections = [anchor_predictions] # Start with original value
        # Removed all_mask_info list

        global_constraint_idx = -1

        for constraint in constraints:
             global_constraint_idx = self.constraints.index(constraint)
             inequality = constraint.single_inequality
             potential_mask_atoms: List[Atom] = [
                 atom for atom in inequality.body if atom.variable.id != anchor_id
             ]

             if not potential_mask_atoms:
                 continue

             batch_size = anchor_predictions.shape[0]
             num_mask_vars = len(potential_mask_atoms)
             attn_scores = torch.zeros((batch_size, num_mask_vars), device=preds.device)
             mask_var_preds = []

             for i, mask_atom in enumerate(potential_mask_atoms):
                 mask_var_id = mask_atom.variable.id
                 h_M = preds[:, mask_var_id]
                 mask_var_preds.append(h_M)
                 attn_net = self.attention_nets[f'attn_score_{global_constraint_idx}']
                 score = attn_net(anchor_predictions, h_M).squeeze(-1)
                 attn_scores[:, i] = score

             attn_weights = F.softmax(attn_scores, dim=1)
             # Removed calculation of winning_mask_id

             h_M_stack = torch.stack(mask_var_preds, dim=1)
             weighted_h_M = (attn_weights.unsqueeze(1) @ h_M_stack.unsqueeze(-1)).squeeze(-1).squeeze(-1)

             correction = weighted_h_M.clone()

             epsilon = torch.full_like(correction, EPSILON)
             strict = inequality.ineq_sign in ['<', '>']
             if inequality.ineq_sign == '<': correction -= epsilon if strict else 0.0
             elif inequality.ineq_sign == '>': correction += epsilon if strict else 0.0

             mask_var_ids = {atom.variable.id for atom in potential_mask_atoms}
             for atom in inequality.body:
                 atom_id = atom.variable.id
                 if atom_id == anchor_id or atom_id in mask_var_ids:
                     continue
                 y_NM = ground_truth[:, atom_id]
                 sign_factor = -1.0 if atom.positive_sign else 1.0
                 correction += y_NM * atom.coefficient * sign_factor

             all_corrections.append(correction)
             # Removed appending to all_mask_info

        all_corrections_tensor = torch.stack(all_corrections, dim=0)
        # Removed all_mask_info_tensor

        # Only need the max value, not the indices for mask gathering
        corrected_anchor_preds, _ = torch.max(all_corrections_tensor, dim=0) # Shape: (batch,)

        # Removed gathering best_mask_info

        return corrected_anchor_preds # Return only the corrected values