from typing import List, Union, Tuple
import torch
import torch.nn.functional as F
from collections import defaultdict
import random

from pishield.linear_requirements.classes import Variable, Constraint, Atom
from pishield.linear_requirements.compute_sets_of_constraints import get_pos_neg_x_constr, compute_sets_of_constraints
from pishield.linear_requirements.correct_predictions import get_constr_at_level_x, get_final_x_correction
from pishield.linear_requirements.feature_orderings import set_ordering
from pishield.linear_requirements.parser import parse_constraints_file, split_constraints, remap_constraint_variables
from pishield.linear_requirements.constants import EPSILON, INFINITY


class ShieldLayer(torch.nn.Module):
    def __init__(self, num_variables: int, requirements_filepath: str, ordering_choice: str = 'given',
                 init_temp: float = 1.0, min_temp: float = 0.1, anneal_rate: float = 0.01):

        super().__init__()
        self.num_variables = num_variables
        ordering, constraints = parse_constraints_file(requirements_filepath)
        # clustered_constraints = split_constraints(ordering, constraints)
        # TODO: Remember that the following optimization is purely required for the
        # supervised learning case (regression)
        ordering, constraints, _ = remap_constraint_variables(ordering, constraints)
        self.ordering = set_ordering(ordering, ordering_choice)
        self.constraints = constraints
        self.sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        self.pos_matrices, self.neg_matrices = self.create_matrices()
        self.dense_ordering = self.get_dense_ordering()  # requires self.sets_of_constraints
        self.masks = {}

        # Parameter dictionary for dynamic learnable params
        self._mask_logit_params = torch.nn.ParameterDict()

        self.register_buffer('temperature', torch.tensor(init_temp))
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate
        # Learnable scale factor for mask gradients
        self.mask_scale = torch.nn.Parameter(torch.tensor(1.0))

        self.B = -1
        self.C = -1
        self.N = -1

    def create_matrices(self):
        # this function creates matrices C+ and C- for each variable x_i
        # note that the column corresponding to x_i in the matrices will be 0s
        pos_matrices: {Variable: torch.Tensor} = {}
        neg_matrices: {Variable: torch.Tensor} = {}
        for x in self.sets_of_constr:
            x: Variable
            # print(x.id)
            x_constr = get_constr_at_level_x(x, self.sets_of_constr)
            pos_x_constr, neg_x_constr = get_pos_neg_x_constr(x, x_constr)

            pos_matrices[x] = self.create_matrix(x, pos_x_constr, positive_x=True)
            neg_matrices[x] = self.create_matrix(x, neg_x_constr, positive_x=False)
        return pos_matrices, neg_matrices

    def get_dense_ordering(self) -> List[Variable]:
        dense_ordering = []
        for x in self.ordering:
            x_constr = get_constr_at_level_x(x, self.sets_of_constr)
            if len(x_constr) == 0:
                continue
            else:
                dense_ordering.append(x)
        return dense_ordering

    def create_matrix(self, x: Variable, x_constr: List[Constraint], positive_x: bool) -> Union[torch.Tensor, float]:
        if len(x_constr) == 0:
            return -INFINITY if positive_x else INFINITY

        matrix = torch.zeros((len(x_constr), self.num_variables), dtype=torch.float)
        x_unsigned_coefficients = torch.ones((len(x_constr),), dtype=torch.float)  # bias (i.e. the constraint constant)
        bias = torch.zeros((len(x_constr),), dtype=torch.float)
        for constr_index, constr in enumerate(x_constr):
            constr: Constraint

            is_strict_inequality = True if constr.single_inequality.ineq_sign == '>' else False
            constant = constr.single_inequality.constant
            epsilon = EPSILON if is_strict_inequality else 0.
            bias[constr_index] = constant + epsilon
            complementary_atoms: List[Atom] = constr.get_body_atoms()
            for atom in complementary_atoms:
                atom_id = atom.variable.id
                if atom_id == x.id:
                    x_unsigned_coefficients[constr_index] = atom.coefficient
                    continue
                else:
                    signed_coefficient = atom.get_signed_coefficient()
                    matrix[constr_index, atom_id] = signed_coefficient

        # next, divide by the unsigned coefficients of x:
        matrix = matrix / x_unsigned_coefficients.unsqueeze(-1)  # num constraints that contain x x num variables

        # if x is positive, multiply by -1 the matrix
        if positive_x:
            matrix *= (-1.)

        # add bias (constraint constant)
        bias = bias / x_unsigned_coefficients
        if not positive_x:
            bias *= (-1.)

        matrix = torch.cat([matrix, bias.unsqueeze(1)], dim=1)
        return matrix

    # def __call__(self, preds, *args, **kwargs):
    def forward(self, preds: torch.Tensor, ground_truth: torch.Tensor | None = None):

        if self.training:
            self.masks = {}

        device = preds.device
        N = preds.shape[-1]
        corrected_preds = torch.cat([preds.clone(), torch.ones(preds.shape[0], 1, device=device)], dim=1)
        preds = corrected_preds.clone()

        if isinstance(ground_truth, torch.Tensor):
            ground_truth = torch.cat([ground_truth.clone(), torch.ones(ground_truth.shape[0], 1, device=device)], dim=1)

        for x in self.dense_ordering:
            pos = x.id

            # So for each variable in the ordering, we need to ensure that all the variables have a probable mask
            # pos_matrix and neg_matrix have shape: num constraints that contain x x num variables

            pos_matrix, pos_masks = self.apply_matrix(preds.clone(), self.pos_matrices[x],
                                                      pos,
                                                      ground_truth=ground_truth,
                                                      reduction='max')
            neg_matrix, neg_masks = self.apply_matrix(preds.clone(), self.neg_matrices[x],
                                                      pos,
                                                      ground_truth=ground_truth,
                                                      reduction='min')

            corrected_preds[:, pos], final_masks = get_final_x_correction(preds[:, pos],
                                                                          pos_matrix,
                                                                          neg_matrix,
                                                                          pos_masks,
                                                                          neg_masks)

            if self.training:
                self.masks[pos] = final_masks

            preds = corrected_preds.clone()
            corrected_preds = preds.clone()
        return corrected_preds[:, :N]

    def apply_matrix(self, preds: torch.Tensor, matrix: Union[torch.Tensor, float],
                     variable_idx: int,
                     ground_truth: torch.Tensor | None = None,
                     reduction='none') -> Tuple[torch.Tensor, torch.Tensor | None]:

        device = preds.device

        if isinstance(matrix, torch.Tensor):
            matrix = matrix.to(device)
        else:
            return matrix, None

        # Failsafe for inference time: During inference you need to ensure that
        # ground truth values aren't used.
        if not self.training:
            ground_truth = None

        self.B = preds.shape[0]  # batch size
        self.C = matrix.shape[0]  # num constraints in the current set
        self.N = matrix.shape[1]  # num variables

        # expand tensors
        preds = preds.unsqueeze(1).expand((self.B, self.C, self.N))

        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.clone().unsqueeze(1).expand((self.B, self.C, self.N)).to(device)

        matrix = matrix.clone().unsqueeze(0).expand((self.B, self.C, self.N))

        # Use the ground truth if given, or else use the default value from before
        # This happens in two cases, during inference, or when using the shield
        # layer without masking of gradients
        selections, masks = self.__apply_mask(preds, matrix, variable_idx,
                                              reduction, ground_truth)
        result = (selections * matrix).sum(dim=2)

        selected_mask, mask_index = None, None

        if reduction == 'max':
            result, mask_index = result.max(dim=1)
        elif reduction == 'min':
            result, mask_index = result.min(dim=1)
        else:
            selected_mask = masks

        # Use the mask indices, which are basically just which constraint the
        # mask was selected from for the given batch item.
        if mask_index is not None and masks is not None:
            batch_idx = torch.arange(self.B, device=device)
            selected_mask = masks[batch_idx, mask_index]

        return result, selected_mask

    def anneal_temperature(self, device):
        self.temperature = torch.max(self.temperature * (1.0 - self.anneal_rate),
                                     torch.tensor(self.min_temp, device=device))

    def __apply_mask(self,
                     preds: torch.Tensor,           # (B, C, N)
                     matrix: torch.Tensor,          # (B, C, N)
                     variable_idx: int,
                     reduction: str,
                     ground_truth: torch.Tensor | None = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:

        # If no ground truth (i.e. inference), skip masking
        if ground_truth is None:
            return preds, None

        # Build eligibility mask: which vars actually appear in each constraint
        eligible = (matrix.abs() > EPSILON)            # (B, C, N) Bool

        # Create or fetch a learnable logit tensor for this (x_i, direction) pair
        key = f"{variable_idx}_{reduction}"
        if key not in self._mask_logit_params:
            # initialize small random logits for each constraint row
            init = torch.zeros(1, matrix.size(1), matrix.size(2), device=preds.device)
            # shape (1, C, N) so we can broadcast over batch
            self._mask_logit_params[key] = torch.nn.Parameter(init)

        logits = self._mask_logit_params[key].expand(preds.size(0), -1, -1)
        # Mask out ineligible positions by setting their logits to -inf
        logits = torch.where(eligible, logits, torch.tensor(-INFINITY, device=logits.device))

        # Soft attention weights over the N variables in each constraint
        # Shape: (B, C, N), sums to 1 along dim=-1
        soft_masks = F.softmax(logits / self.temperature, dim=-1)

        # If you want a “hard” mask at inference, you can:
        # hard_idx = soft_masks.argmax(dim=-1)        # (B, C)
        # then build a one-hot from that. But during training we keep it soft.

        # Mix preds and ground_truth per-variable via soft mask:
        # for each sample b, constraint c, variable j:
        #   selection = soft_masks[b,c,j]*preds[b,c,j] + (1-soft_masks[b,c,j])*gt[b,c,j]
        selections = soft_masks * preds + (1.0 - soft_masks) * ground_truth

        # `selections` flows gradients to preds weighted by soft_masks,
        # yet still anchors all other coords toward ground_truth in proportion.

        return selections, soft_masks

