from typing import (
    Literal
)
from torch import (
    Tensor, pi, exp, sum, log,
    squeeze, diagonal, maximum,
    minimum, zeros_like
)
from torch.linalg import (
    inv, det
)


def elk_mat_gauss(
        mu: Tensor,
        sigma: Tensor
) -> Tensor:
    """
    Calculate Expected Likelihood Kernel (ELK) matrix of pairwise elements in a mixture
    of Gaussian distribution. The mixture is composed of n Gaussian elements, and
    each entry [i][j] in the returned matrix represents ELK(p_i, p_j), where p_i and
    p_j are ith and jth elements in the distribution.
    For Gaussian elements:
            ELK(p_i, p_j) = q_ji(mu_i)
    where:
            q_ji = N(mu_j, sigma_i + sigma_j)
    mu_* and sigma_* are mean vector and covariance matrix of an element respectively.

    :param mu: torch tensor, shape of [b, n, d], batch of b stacks of mean
               vectors of n Gaussian elements in d dimensions.
    :param sigma: torch tensor, shape of [b, n, d, d], n stacked covariance
                  matrices in d dimensions, for a batch of b Gaussian mixtures.
    :return: torch tensor, shape of [b, n, n], matrices of pairwise p_i(mu_j)
             for a batch of b mixtures, each entry [i][j] represents p_i(mu_j).
    """
    d = mu.shape[-1]
    sigma_sum = sigma[:, None, ...] + sigma[:, :, None, ...]  # [b, n, n, d]
    mu_sub = mu[:, None, ...] - mu[:, :, None, ...]  # [b, n, n, d]
    numerator = squeeze(exp(-0.5 * (mu_sub[..., None, :] @ inv(sigma_sum)) @ mu_sub[..., None]))  # [n, n]
    denominator = (((2 * pi) ** d) * det(sigma_sum)) ** 0.5  # [n, n]
    elk_mat = numerator / denominator
    return elk_mat


def elk_mat_uniform(
        mu: Tensor,
        sigma: Tensor
) -> Tensor:
    """
    Calculate Expected Likelihood Kernel (ELK) matrix of pairwise elements in a mixture
    of uniform distribution. The mixture is composed of n uniform elements, and
    each entry [i][j] in the returned matrix represents ELK(p_i, p_j), where p_i and
    p_j are ith and jth elements in the distribution.
    For uniform elements,
            ELK(p_i, p_j) = (V_i∩j) / (V_i * V_j)
    where V_* is the "volume" of hyper rectangle of that element.

    :param mu: torch tensor, shape of [b, n, d], batch of b centre values of
               n uniform elements of d dimensions.
    :param sigma: torch tensor, shape of [b, n, d, d], batch of b stacks of n
                  diagonal matrices of size d, each diagonal entry indicates
                  the interval length of an element on that dimension.
    :return: torch tensor, shape of [b, n, n], batch of b matrices of pairwise
             p_i(mu_j), each entry [i][j] represents p_i(mu_j).
    """
    half_interval = 0.5 * diagonal(sigma, dim1=2, dim2=3)  # [b, n, d]
    log_volume = sum(log(half_interval * 2), dim=2)  # [b, n]
    log_volume_prod = log_volume[:, None, ...] + log_volume[:, :, None, ...]  # [b, n, n, d]
    upper = mu + half_interval  # [b, n, d]
    lower = mu - half_interval  # [b, n, d]
    overlap_lower = maximum(lower[:, None, ...], lower[:, :, None, ...])  # [b, n, n, d]
    overlap_upper = minimum(upper[:, None, ...], upper[:, :, None, ...])  # [b, n, n, d]
    log_overlap_volume = sum(log(maximum(overlap_upper - overlap_lower, zeros_like(overlap_lower))), dim=-1)
    elk_mat = exp(log_overlap_volume - log_volume_prod)
    return elk_mat


def estimate(
        c: Tensor,
        mu: Tensor,
        sigma: Tensor,
        elem: Literal["gauss", "uniform"],
        **kwargs
):
    r"""
    Pytorch implementation of Expected Likelihood kernel (ELK) entropy estimator
    for estimating entropy of a mixture distribution. Entropy estimated by ELK
    estimator is a lower bound of the true entropy.
    Elements of the mixture could be "gauss" for Gaussian or "uniform" for uniform.
    :param c: torch tensor, shape of [b, n], weights of n elements of batch b,
              should sum up to one on each batch.
    :param mu: torch tensor, shape of [b, n, d], indicates center of n elements
               of dimension d, of batch b.
    :param sigma: torch tensor, shape of [b, n, d, d], represents the "spread" of
                  n elements of dimension d, of b mixtures.
                  If elem='gauss', sigma is n covariance matrices, thus have to be semi-definite;
                  If elem='uniform', sigma is n diagonal matrices, each entry on diagonal indicates
                  the interval on corresponding dimension, e.g., sigma[n][i][i] indicates
                  the interval of nth element on ith dimension.
    :param elem: str, optional, 'gauss' or 'uniform', indicates the element type for the
                 mixture distribution.
    :param kwargs: not used.
    :return: torch tensor, shape of [b,], batch of entropies estimated.
    """
    if elem == 'gauss':
        elk_mat = elk_mat_gauss(mu, sigma)
    elif elem == 'uniform':
        elk_mat = elk_mat_uniform(mu, sigma)
    else:
        raise NotImplementedError(f'Mixture of "{elem}" is not implemented.')
    ent_est = -sum(c * log(sum(c[:, :, None, ...] * elk_mat, dim=1)), dim=1)
    return ent_est
