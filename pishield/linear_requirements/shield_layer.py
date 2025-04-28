from typing import List, Union, Tuple
import torch
from collections import defaultdict
import random

from pishield.linear_requirements.classes import Variable, Constraint, Atom
from pishield.linear_requirements.compute_sets_of_constraints import get_pos_neg_x_constr, compute_sets_of_constraints
from pishield.linear_requirements.correct_predictions import get_constr_at_level_x, get_final_x_correction
from pishield.linear_requirements.feature_orderings import set_ordering
from pishield.linear_requirements.parser import parse_constraints_file, split_constraints, remap_constraint_variables
from pishield.linear_requirements.constants import EPSILON, INFINITY


class ShieldLayer(torch.nn.Module):
    def __init__(self, num_variables: int, requirements_filepath: str, ordering_choice: str = 'given'):
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
    def __call__(self, preds: torch.Tensor,
                 ground_truth: torch.Tensor | None = None):

        device = preds.device
        N = preds.shape[-1]
        corrected_preds = torch.cat([preds.clone(), torch.ones(preds.shape[0], 1, device=device)], dim=1)
        preds = corrected_preds.clone()

        for x in self.dense_ordering:
            pos = x.id

            # So for each variable in the ordering, we need to ensure that all the variables have a probable mask
            # pos_matrix and neg_matrix have shape: num constraints that contain x x num variables

            pos_matrix, pos_masks = self.apply_matrix(preds.clone(), self.pos_matrices[x],
                                           reduction='max',
                                           ground_truth=ground_truth)
            neg_matrix, neg_masks = self.apply_matrix(preds.clone(), self.neg_matrices[x],
                                           reduction='min',
                                           ground_truth=ground_truth)

            corrected_preds[:, pos], final_masks = get_final_x_correction(preds[:, pos],
                                                                          pos_matrix,
                                                                          neg_matrix,
                                                                          pos_masks,
                                                                          neg_masks)

            self.masks[pos] = final_masks

            preds = corrected_preds.clone()
            corrected_preds = preds.clone()
        return corrected_preds[:, :N]

    def apply_matrix(self, preds: torch.Tensor, matrix: Union[torch.Tensor, float],
                     ground_truth: torch.Tensor | None = None,
                     reduction='none') -> Tuple[torch.Tensor, torch.Tensor | None]:

        # Failsafe for inference time: During inference you need to ensure that
        # ground truth values aren't used.
        if not self.training:
            ground_truth = None

        if type(matrix) != torch.Tensor:
            return matrix, None
        else:
            matrix = matrix.to(preds.device)

        self.B = preds.shape[0]  # batch size
        self.C = matrix.shape[0]  # num constraints in the current set
        self.N = matrix.shape[1]  # num variables

        # expand tensors
        preds = preds.unsqueeze(1).expand((self.B, self.C, self.N))

        if ground_truth is not None:
            ground_truth = ground_truth.clone().unsqueeze(1).expand((self.B, self.C, self.N)).to(preds.device)

        matrix = matrix.clone().unsqueeze(0).expand((self.B, self.C, self.N))

        # Use the ground truth if given, or else use the default value from before
        # This happens in two cases, during inference, or when using the shield
        # layer without masking of gradients
        selections, masks = self.__apply_mask(preds, matrix, ground_truth)
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
            batch_idx = torch.arange(self.B, device=preds.device)
            selected_mask = masks[batch_idx, mask_index]

        return result, selected_mask

    def __apply_mask(self, preds: torch.Tensor,
                     matrix: torch.Tensor,
                     ground_truth: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # This function basically returns for each constraint, the potential mask indices that can be used
        # Should be used for every element in the batch
        # This function assumes we are working only with matrices that encode constraints

        # In case ground truth not given, return default values
        if ground_truth is None:
            return preds, None

        mask_indices = (torch.abs(matrix) > EPSILON).nonzero()

        mask_groupings = defaultdict(lambda: defaultdict(list))
        for batch, constr_idx, var_idx in mask_indices:
            mask_groupings[batch.item()][constr_idx.item()].append(var_idx.item())

        masks = torch.zeros(self.B, self.C, self.N, dtype=torch.bool, device=preds.device)
        for b in range(self.B):
            if b in mask_groupings:
                for c, grouping in mask_groupings[b].items():
                    if grouping:
                        # Select any one of the possible variables used in the bound constraints
                        # as the mask variable
                        masks[b, c, random.choice(grouping)] = True

        # At the mask indices, select from the prediction, in all other indices, select from the ground truth.
        # The reason this works is because the matrix will zero out all the irrelevant indices (which correspond
        # to variables that are not involved in the inequalities).
        selections = torch.where(masks, preds, ground_truth)
        return selections, masks
