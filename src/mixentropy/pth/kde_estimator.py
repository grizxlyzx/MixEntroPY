from typing import (
    Literal
)
from torch import (
    Tensor, pi, exp, sum, log,
    prod, diagonal, squeeze, all,
    transpose
)
from torch.linalg import (
    inv, det
)


def prob_mean_mat_gauss(
        mu: Tensor,
        sigma: Tensor
) -> Tensor:
    r"""
    Calculate batch of p_i(mu_j) matrix of pairwise elements in a mixture
    of Gaussian distribution. Each mixture is composed of n Gaussian
    elements, and each entry [i][j] in the returned matrix represents
    pdf_i(mu_j), where pdf_i and mu_j are pdf of ith element and mean
    of jth element in the distribution, which is equivalent as
    calculating K_j(x_i - x_j) where K is a Gaussian kernel.
    :param mu: torch tensor, shape of [b, n, d], batch of b stacks of mean
               vectors of n Gaussian elements in d dimensions.
    :param sigma: torch tensor, shape of [b, n, d, d], n stacked covariance
                  matrices in d dimensions, for a batch of b Gaussian mixtures.
    :return: torch tensor, shape of [b, n, n], matrices of pairwise p_i(mu_j)
             for a batch of b mixtures, each entry [i][j] represents p_i(mu_j).
    """
    d = mu.shape[-1]
    mu_sub = mu[:, None, ...] - mu[:, :, None, ...]  # [b, n, n, d]
    numerator = transpose(
        squeeze(exp(-0.5 * (mu_sub @ inv(sigma))[..., None, :] @ mu_sub[..., None])), dim0=1, dim1=2
    )  # [b, n, n]
    denominator = (((2 * pi) ** d) * det(sigma)) ** 0.5
    prob_mean_mat = transpose(numerator / denominator[:, None, ...], dim0=1, dim1=2)  # [i][j] --> p_i(mu_j)
    return prob_mean_mat


def prob_mean_mat_uniform(
        mu: Tensor,
        sigma: Tensor
) -> Tensor:
    r"""
    Calculate p_i(mu_j) matrix of pairwise elements in a batch of mixture
    of uniform distribution. Each mixture is composed of n uniform
    elements, and each entry [i][j] in the returned matrix represents
    pdf_i(mu_j), where pdf_i and mu_j are pdf of ith element and mean
    of jth element in the distribution, which is equivalent as
    calculating K_j(x_i - x_j) where K is a uniform kernel.
    Notice that the gradiant of components' means w.r.t pdf are all zeros,
    thus using entropy estimated by KDE estimator with uniform elements
    as a part of the loss function is NOT appropriate.
    :param mu: torch tensor, shape of [b, n, d], batch of b centre values of
               n uniform elements of d dimensions.
    :param sigma: torch tensor, shape of [b, n, d, d], batch of b stacks of n
                  diagonal matrices of size d, each diagonal entry indicates
                  the interval length of an element on that dimension.
    :return: torch tensor, shape of [b, n, n], batch of b matrices of pairwise
             p_i(mu_j), each entry [i][j] represents p_i(mu_j).
    """
    half_interval = 0.5 * diagonal(sigma, dim1=2, dim2=3)  # [n, d]
    upper = mu + half_interval  # [b, n, d]
    lower = mu - half_interval  # [b, n, d]
    p = 1. / prod(half_interval * 2, dim=2)  # [b, n]
    dst_upper = mu[:, None, ...] - upper[:, :, None, ...]
    dst_lower = mu[:, None, ...] - lower[:, :, None, ...]
    prob_mean_mat = all((dst_upper <= 0) * (dst_lower >= 0), dim=-1)  # [n, n], [i][j] --> p_i(mu_j) != 0
    prob_mean_mat = prob_mean_mat * p[:, :, None, ...]
    return prob_mean_mat


def estimate(
        c: Tensor,
        mu: Tensor,
        sigma: Tensor,
        elem: Literal["gauss", "uniform"] = 'gauss',
        **kwargs
) -> Tensor:
    r"""
    Pytorch implementation of KDE entropy estimator for mixture distribution
    using the probability of the component means. When estimating mixture of
    Gaussian kernels share the same covariance matrix, estimation is equivalent
    to pairwise KL divergence estimation minus 2/d, thus this estimator could be
    used as an upper bound of true entropy as long as 2/d is added to the
    estimation, where d is the number of dimension.
    NOTICE: When estimating entropy of the mixture of uniform, the gradiant
    of each components' position w.r.t entropy estimated are all zeros, since the
    PDF of a multivariate uniform distribution is a hyper rectangle, which is pure
    flat, resulting in no direction to which the mean move towards could change
    the estimated entropy smoothly. Thus using entropy estimated by KDE estimator
    with uniform elements as a part of the loss function is NOT appropriate.
    :param c: torch tensor, shape of [b, n], weights of n elements of batch b,
              should sum up to one on each batch.
    :param mu: torch tensor, shape of [b, n, d], indicates center of n elements
               of dimension d, of batch b.
    :param sigma: torch tensor, shape of [b, n, d, d], represents the "spread" of
                  n elements of dimension d, of batch b.
                  If elem='gauss', sigma is batch of n covariance matrices, thus
                    every matrix have to be semi-definite;
                  If elem='uniform', sigma is a batch of n diagonal matrices, each
                    entry on diagonal indicates the interval on corresponding
                    dimension, e.g., sigma[n][i][i] indicates the interval of nth
                    element on ith dimension.
    :param elem: str, optional, 'gauss' or 'uniform', indicates the element type
                 for the mixture distribution.
    :param kwargs: not used.
    :return: torch tensor, shape of [b,], batch of entropies estimated.
    """
    if elem == 'gauss':
        prob_mean_mat = prob_mean_mat_gauss(mu, sigma)
    elif elem == 'uniform':
        prob_mean_mat = prob_mean_mat_uniform(mu, sigma)
    else:
        raise NotImplementedError(f'Mixture of "{elem}" is not implemented.')
    ent_est = -sum(c * log(sum(c[:, :, None, ...] * prob_mean_mat, dim=1)), dim=1)
    return ent_est
