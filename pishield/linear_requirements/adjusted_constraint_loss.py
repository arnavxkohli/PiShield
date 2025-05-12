import torch

def adjusted_constraint_loss(
    output: torch.Tensor,      # (B, N)
    target: torch.Tensor,      # (B, N)
    masks: dict[int, torch.Tensor | None],
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    If a mask for var i is:
      • None  -> use plain MSE on column i
      • bool  -> zero‐out misaligned samples (hard‐mask case)
      • float -> re‐weight squared‐error by (1 + mask_weight)
    """
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError(f"Invalid reduction: {reduction!r}")

    errs = output - target          # (B, N)
    sq   = errs.pow(2)              # (B, N)
    signs = torch.sign(errs)        # (B, N)

    B, N = output.shape
    device = output.device

    for var_idx, mask in masks.items():
        if mask is None:
            continue

        mask = mask.to(device)
        # boolean mask: zero‐out misaligned
        if mask.dtype == torch.bool:
            # find samples where mask[b,j] is True, then check sign alignment
            coords = mask.nonzero(as_tuple=False)
            b_idx, j_idx = coords[:,0], coords[:,1]
            var_signs  = signs[b_idx, var_idx]
            mask_signs = signs[b_idx, j_idx]
            aligned = (var_signs * mask_signs > 0).to(dtype=sq.dtype)
            sq[b_idx, var_idx] *= aligned

        # float mask: re‐weight
        else:
            # weight = 1 + mask_weight on that var
            w = 1.0 + mask[:, var_idx]
            sq[:, var_idx] *= w

    if reduction == 'mean':
        return sq.mean()
    elif reduction == 'sum':
        return sq.sum()
    return sq
