import pickle as pkl
import torch
from typing import Union

INFINITY = torch.inf


def get_constr_at_level_x(x, sets_of_constr):
    for var in sets_of_constr:
        if var.id == x.id:
            return sets_of_constr[var]


def get_final_x_correction(
    initial_vals:      torch.Tensor,
    pos_corrected:     Union[torch.Tensor, float, None],
    neg_corrected:     Union[torch.Tensor, float, None],
    pos_mask_weights:  Union[torch.Tensor, None],
    neg_mask_weights:  Union[torch.Tensor, None],
) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """
    Returns:
      - corrected_vals: Tensor (B,)
      - final_masks:    None, or Tensor (B,N) of either bools (hard) or floats (soft)
    """
    B = initial_vals.size(0)
    device = initial_vals.device

    # Determine number of variables N from whichever mask is present
    N = None
    if isinstance(pos_mask_weights, torch.Tensor):
        N = pos_mask_weights.size(1)
    elif isinstance(neg_mask_weights, torch.Tensor):
        N = neg_mask_weights.size(1)

    # Helper to check “is a Tensor with real data”
    def is_real_tensor(x):
        return isinstance(x, torch.Tensor)

    # -----------------------------------------
    # 1) HARD-MASK (fallback) if either mask is None
    # -----------------------------------------
    if pos_mask_weights is None or neg_mask_weights is None:
        hard_mask = None

        # Positive pass
        if is_real_tensor(pos_corrected):
            safe_pos = pos_corrected.where(~pos_corrected.isinf(), initial_vals)
            tmp_vals, chose_pos = torch.stack([initial_vals, safe_pos], dim=1).max(dim=1)
            if N is not None:
                hard_mask = chose_pos.unsqueeze(1).expand(B, N) & pos_mask_weights.bool()
        else:
            tmp_vals = initial_vals

        # Negative pass
        if is_real_tensor(neg_corrected):
            safe_neg = neg_corrected.where(~neg_corrected.isinf(), initial_vals)
            final_vals, chose_neg = torch.stack([tmp_vals, safe_neg], dim=1).min(dim=1)
            if hard_mask is not None:
                hard_mask = torch.where(
                    chose_neg.unsqueeze(1).expand(B, N),
                    neg_mask_weights.bool(),
                    hard_mask
                )
        else:
            final_vals = tmp_vals

        return final_vals, (hard_mask if hard_mask is not None else None)

    # -----------------------------------------
    # 2) SOFT-MASK (training)  
    # -----------------------------------------
    # Initialize a float accumulator
    final_mask_weights = torch.zeros((B, N), device=device, dtype=torch.float32)

    # Positive pass
    safe_pos = (
        pos_corrected.where(~pos_corrected.isinf(), initial_vals)
        if is_real_tensor(pos_corrected)
        else initial_vals
    )
    tmp_vals, chose_pos = torch.stack([initial_vals, safe_pos], dim=1).max(dim=1)
    chose_pos_f = chose_pos.to(torch.float32).unsqueeze(1)  # (B,1)
    final_mask_weights += chose_pos_f * pos_mask_weights

    # Negative pass
    safe_neg = (
        neg_corrected.where(~neg_corrected.isinf(), initial_vals)
        if is_real_tensor(neg_corrected)
        else tmp_vals
    )
    final_vals, chose_neg = torch.stack([tmp_vals, safe_neg], dim=1).min(dim=1)
    chose_neg_f = chose_neg.to(torch.float32).unsqueeze(1)
    final_mask_weights += chose_neg_f * neg_mask_weights

    return final_vals, final_mask_weights


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

