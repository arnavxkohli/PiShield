from typing import List
import torch
import logging

logger = logging.getLogger(__name__)


def eval_atoms_list(atoms_list: List, preds: torch.Tensor, reduction='sum'):
    evaluated_atoms = []
    for atom in atoms_list:
        atom_value = preds[:, atom.variable.id]
        evaluated_atoms.append(atom.eval(atom_value))

    if evaluated_atoms == []:
        return torch.zeros(preds.shape[0])

    evaluated_atoms = torch.stack(evaluated_atoms, dim=1)
    if reduction == 'sum':
        result = evaluated_atoms.sum(dim=1)
    else:
        raise Exception(f'{reduction} reduction not implemented!')
    return result


def check_constraint_satisfaction(preds: torch.Tensor, constraints: List) -> bool:
    all_constr_sat = True
    for constr in constraints:
        sat = constr.check_satisfaction_per_sample(preds)
        if not sat.all():
            all_constr_sat = False
            # raise Exception('Not satisfied!', constr.readable())
            logger.debug(f'Not satisfied! Constraint: {constr.readable()}')
        else:
            logger.debug(f"Constraint {constr.readable()} satisfied.")
    if all_constr_sat:
        logger.info('All constraints are satisfied!')
    return all_constr_sat
