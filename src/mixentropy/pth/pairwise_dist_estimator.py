from typing import (
    Literal
)
from torch import (
    Tensor, tensor, pi, sum, log, logdet,
    exp, inf, squeeze, diagonal, maximum,
    minimum, all, transpose, zeros_like
)
from torch.linalg import (
    inv
)

pi = tensor(pi)
trace = lambda x, dim1, dim2: sum(diagonal(x, dim1=dim1, dim2=dim2), dim=-1)


def div_mat_kl_gaussian(
        mu: Tensor,
        sigma: Tensor,
        d: int,
        **kwargs
) -> Tensor:
    inv_sigma = inv(sigma)  # [b, n, d, d]
    ld_sigma = logdet(sigma)  # [b, n]
    mu_sub = mu[:, :, None, ...] - mu[:, None, ...]  # mu2 - mu1, [b, n, n, d]
    div_mat = inv_sigma[:, None, ...] @ sigma[:, :, None, ...]
    div_mat = trace(div_mat, dim1=-2, dim2=-1)  # [b, n, n]
    div_mat += -d
    div_mat += transpose(squeeze((mu_sub @ inv_sigma)[..., None, :] @ mu_sub[..., None]), dim0=-1, dim1=-2)
    div_mat += ld_sigma[:, None, ...] - ld_sigma[:, :, None, ...]
    div_mat *= 0.5
    return div_mat


def div_mat_kl_uniform(
        mu: Tensor,
        sigma: Tensor,
        **kwargs
) -> Tensor:
    # if two uniform element have some non-overlap region,
    # KL divergence between them is positive infinity.
    # check if non-overlapping region exist
    half_interval = 0.5 * diagonal(sigma, dim1=2, dim2=3)  # [b, n, d]
    log_volume = sum(log(half_interval * 2), dim=2)
    upper = mu + half_interval  # [n, d]
    lower = mu - half_interval  # [n, d]
    dst_upper = upper[:, None, ...] - upper[:, :, None, ...]  # [b, n, n, d]
    dst_lower = lower[:, None, ...] - lower[:, :, None, ...]  # [b, n, n, d]
    contains = all(
        (dst_upper <= zeros_like(dst_upper)) * (dst_lower >= zeros_like(dst_lower)),
        dim=-1
    )  # [n, n], [i][j] --> p_i contains p_j
    div_mat = (log_volume[:, :, None, ...] - log_volume[:, None, ...]) * contains
    div_mat[~contains] = inf
    return div_mat


def div_mat_chernoff_alpha_gaussian(
        mu: Tensor,
        sigma: Tensor,
        alpha: float = 0.5,
        **kwargs
) -> Tensor:
    mu_sub = mu[:, :, None, ...] - mu[:, None, ...]
    weighted_sigma = ((1 - alpha) * sigma)[:, None, ...] + (alpha * sigma)[:, :, None, ...]
    ld_sigma = logdet(sigma)
    ld_weighted_sigma = logdet(weighted_sigma)
    div_mat = squeeze((mu_sub[..., None, :] @ inv(weighted_sigma)) @ mu_sub[..., None])
    div_mat *= ((1 - alpha) * alpha)
    div_mat += ld_weighted_sigma - (((1 - alpha) * ld_sigma)[..., None] + (alpha * ld_sigma)[..., None, :])
    div_mat *= 0.5
    return div_mat


def div_mat_chernoff_alpha_uniform(
        mu: Tensor,
        sigma: Tensor,
        alpha: float = 0.5,
        **kwargs
) -> Tensor:
    half_interval = 0.5 * diagonal(sigma, dim1=2, dim2=3)  # [b, n, d]
    log_volume = sum(log(half_interval * 2), dim=2)  # [b, n]
    upper = mu + half_interval  # [b, n, d]
    lower = mu - half_interval  # [b, n, d]
    overlap_lower = maximum(lower[:, None, ...], lower[:, :, None, ...])  # [b, n, n, d]
    overlap_upper = minimum(upper[:, None, ...], upper[:, :, None, ...])  # [b, n, n, d]
    log_overlap_volume = sum(log(maximum(overlap_upper - overlap_lower, zeros_like(overlap_upper))), dim=3)  # [b, n, n]
    div_mat = alpha * log_volume[:, None, ...] + (1 - alpha) * log_volume[:, :, None, ...] - log_overlap_volume
    return div_mat


def estimate(
        c: Tensor,
        mu: Tensor,
        sigma: Tensor,
        elem: Literal["gauss", "uniform"] = 'gauss',
        dist: Literal["kl", "chernoff"] = 'kl',
        **kwargs
) -> Tensor:
    if elem == 'gauss':
        _, n, d = mu.shape
        ent_x = 0.5 * (logdet(sigma) + d * log(2 * pi) + d)  # calculate individual entropies
        if dist == 'kl':
            div_mat = div_mat_kl_gaussian(mu, sigma, d, **kwargs)
        elif dist == 'chernoff':
            div_mat = div_mat_chernoff_alpha_gaussian(mu, sigma, **kwargs)
        else:
            raise NotImplementedError(f'Distance function: "{dist}" is not implemented')

    elif elem == 'uniform':
        ent_x = sum(log(diagonal(sigma, dim1=2, dim2=3)), dim=2)  # calculate individual entropies
        if dist == 'kl':
            div_mat = div_mat_kl_uniform(mu, sigma, **kwargs)
        elif dist == 'chernoff':
            div_mat = div_mat_chernoff_alpha_uniform(mu, sigma, **kwargs)
        else:
            raise NotImplementedError(f'Distance function: "{dist}" is not implemented')
    else:
        raise NotImplementedError(f'Mixture of "{elem}" is not implemented.')

    ent_xc = sum(c * ent_x, dim=1)
    ent_est = c[:, None, ...] * exp(-div_mat)
    ent_est = c * log(sum(ent_est, dim=2))
    ent_est = ent_xc - sum(ent_est, dim=1)
    return ent_est
