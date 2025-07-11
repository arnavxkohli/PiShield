import pickle as pkl
import torch

INFINITY = torch.inf


def get_constr_at_level_x(x, sets_of_constr):
    for var in sets_of_constr:
        if var.id == x.id:
            return sets_of_constr[var]


def get_final_x_correction(initial_x_val: torch.Tensor, pos_x_corrected: torch.Tensor,
                           neg_x_corrected: torch.Tensor,
                           pos_masks: torch.Tensor | None = None,
                           neg_masks: torch.Tensor | None = None) -> torch.Tensor:

    # Masks Shape: batch_size x num_variables, X Shape: batch_size
    # The modification to this function requires selecting the correct mask.
    # Three options: Select the unchanged mask (so the mask with all False rows)
    # Select the positive mask (for some batch items)
    # Select the negative mask (for some batch items)
    # This depends on which value of the max is being chosen as well.

    B = initial_x_val.shape[0]
    device = initial_x_val.device

    N = -1
    if isinstance(pos_masks, torch.Tensor):
        N = pos_masks.shape[1]
    elif isinstance(neg_masks, torch.Tensor):
        N = neg_masks.shape[1]

    current_mask = None
    if N != -1:
        current_mask = torch.zeros((B, N), dtype=torch.bool, device=device)

    if not isinstance(pos_x_corrected, torch.Tensor):
        result_1 = initial_x_val
    else:
        pos_x_corrected = pos_x_corrected.where(~(pos_x_corrected.isinf()), initial_x_val)
        result_1, indices_1 = torch.cat([initial_x_val.unsqueeze(1), pos_x_corrected.unsqueeze(1)], dim=1).max(dim=1)
        if current_mask is not None:
            current_mask = torch.where(indices_1.unsqueeze(1).bool(), pos_masks, current_mask)

    if not isinstance(neg_x_corrected, torch.Tensor):
        result_2 = result_1
    else:
        # keep_initial_neg_mask = neg_x_corrected.isinf()
        # neg_x_corrected[keep_initial_neg_mask] = initial_x_val[keep_initial_neg_mask]
        neg_x_corrected = neg_x_corrected.where(~(neg_x_corrected.isinf()), initial_x_val)
        result_2, indices_2 = torch.cat([result_1.unsqueeze(1), neg_x_corrected.unsqueeze(1)], dim=1).min(dim=1)
        if current_mask is not None:
            current_mask = torch.where(indices_2.unsqueeze(1).bool(), neg_masks, current_mask)

    # Remove Bias column when returning the mask
    return result_2, (current_mask[:, :-1] if isinstance(current_mask, torch.Tensor) else current_mask)


def example_predictions_heloc():
    # data = pd.read_csv(f"../data/heloc/test_data.csv")
    # data = data.to_numpy().astype(float)
    # return torch.tensor(data)

    data = pkl.load(open('/home/mihian/DEL_unsat/TEMP_uncons.pkl', 'rb'))
    return data


def check_all_constraints_are_sat(constraints, preds, corrected_preds, verbose=False):
    # print('sat req?:')
    for constr in constraints:
        sat = constr.check_satisfaction(preds)
        if not sat and verbose:
            print('Not satisfied!', constr.readable())

    # print('*' * 80)
    all_sat_after_correction = True
    for constr in constraints:
        sat = constr.check_satisfaction(corrected_preds)
        if not sat:
            all_sat_after_correction = False
            # print('Not satisfied!', constr.readable())
    if all_sat_after_correction and verbose:
        print('All constraints are satisfied after correction!')
    if not all_sat_after_correction:
        print('There are still constraint violations!!!')
        # with open('./TEMP_uncons.pkl', 'wb') as f:
        #     pkl.dump(preds, f, -1)
        # with open('./TEMP_cons.pkl', 'wb') as f:
        #     pkl.dump(corrected_preds, f, -1)
    return all_sat_after_correction


def check_all_constraints_sat(corrected_preds, constraints, error_raise=True):
    all_sat_after_correction = True
    for constr in constraints:
        sat = constr.check_satisfaction_per_sample(corrected_preds)
        if not sat.all():
            # all_sat_after_correction = False
            print(corrected_preds[~sat], 'aaa')
            sample_sat, eval_body_value, constant, ineq_sign = constr.detailed_sample_sat_check(corrected_preds)
            print(sample_sat.all(), eval_body_value[~sample_sat], constant, ineq_sign)
            raise Exception('Not satisfied!', constr.readable())
    return all_sat_after_correction

