from typing import List, Union
import torch
import random

from pishield.linear_requirements.classes import Variable, Constraint, Atom
from pishield.linear_requirements.compute_sets_of_constraints import get_pos_neg_x_constr, compute_sets_of_constraints
from pishield.linear_requirements.correct_predictions import get_constr_at_level_x, get_final_x_correction
from pishield.linear_requirements.feature_orderings import set_ordering
from pishield.linear_requirements.parser import parse_constraints_file, split_constraints

INFINITY = torch.inf
EPSILON = 1e-12


class ShieldLayer(torch.nn.Module):
    def __init__(self, num_variables: int, requirements_filepath: str, ordering_choice: str = 'given'):
        super().__init__()
        self.num_variables = num_variables
        ordering, constraints = parse_constraints_file(requirements_filepath)
        # clustered_constraints = split_constraints(ordering, constraints)
        self.ordering = set_ordering(ordering, ordering_choice)
        self.constraints = constraints
        self.sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        self.pos_matrices, self.neg_matrices = self.create_matrices()
        self.dense_ordering = self.get_dense_ordering()  # requires self.sets_of_constraints

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
    def __call__(self, preds: torch.Tensor):
        device = preds.device
        N = preds.shape[-1]
        corrected_preds = torch.cat([preds.clone(), torch.ones(preds.shape[0], 1, device=device)], dim=1)
        preds = corrected_preds.clone()

        for x in self.dense_ordering:
            pos = x.id

            # pos_matrix and neg_matrix have shape: num constraints that contain x x num variables
            pos_matrix = self.apply_matrix(preds.clone(), self.pos_matrices[x], reduction='amax')
            neg_matrix = self.apply_matrix(preds.clone(), self.neg_matrices[x], reduction='amin')

            corrected_preds[:, pos] = get_final_x_correction(preds[:, pos], pos_matrix, neg_matrix)
            preds = corrected_preds.clone()
            corrected_preds = preds.clone()
        return corrected_preds[:, :N]

    def apply_matrix(self, preds: torch.Tensor, matrix: Union[torch.Tensor, float], reduction='none') -> torch.Tensor:
        if type(matrix) != torch.Tensor:
            return matrix
        else:
            matrix = matrix.to(preds.device)

        B = preds.shape[0]  # batch size
        C = matrix.shape[0]  # num constraints in the current set
        N = matrix.shape[1]  # num variables

        # expand tensors
        preds = preds.unsqueeze(1).expand((B, C, N))
        matrix = matrix.clone().unsqueeze(0).expand((B, C, N))

        result = (preds * matrix).sum(dim=2)
        if reduction == 'amax':
            result = result.amax(dim=1)
        elif reduction == 'amin':
            result = result.amin(dim=1)
        else:
            pass
        return result

    def optimize_gradients(self, preds: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Optimizes gradients based on constraints and current predictions.
        Args:
            preds (torch.Tensor): Current model predictions
            ground_truth (torch.Tensor): Ground truth labels
        Returns:
            torch.Tensor: Modified gradients
        """
        optimized_preds = preds.clone()

        # Group constraints by anchor variable
        constraint_groups = {}
        for constraint in self.constraints:
            anchor = constraint.single_inequality.body[0].variable
            constraint_groups.setdefault(anchor, []).append(constraint)

        # Process each anchor variable
        for anchor, constraints in constraint_groups.items():
            anchor_id = anchor.id
            candidates = []

            for constraint in constraints:
                inequality = constraint.single_inequality

                # Select random atom to mask (one-hot encoding) TODO: Try other methods other than random
                mask_atom = inequality.body[random.randint(0, len(inequality.body) - 1)]
                mask_var_id = mask_atom.variable.id

                # Apply the mask and calculate correction
                correction = optimized_preds[:, mask_var_id].clone()

                # Apply corrections from other atoms
                for atom in inequality.body:
                    if atom.variable.id == anchor_id:
                        continue

                    sign_factor = -1 if atom.positive_sign else 1
                    correction += optimized_preds[:, atom.variable.id] * atom.coefficient * sign_factor

                candidates.append(correction)

            # Update with maximum value from candidates
            if candidates:
                optimized_preds[:, anchor_id] = torch.stack(candidates).amax(dim=0)

        return optimized_preds
