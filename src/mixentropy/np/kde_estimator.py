from typing import (
    Literal
)
from numpy.typing import (
    NDArray
)
from numpy import (
    newaxis, pi, exp, sum, log,
    prod, diagonal, squeeze, all
)
from numpy.linalg import (
    inv, det
)


def prob_mean_mat_gauss(
        mu: NDArray,
        sigma: NDArray
) -> NDArray:
    """
    Calculate p_i(mu_j) matrix of pairwise elements in a mixture
    of Gaussian distribution. The mixture is composed of n Gaussian
    elements, and each entry [i][j] in the returned matrix represents
    pdf_i(mu_j), where pdf_i and mu_j are pdf of ith element and mean
    of jth element in the distribution, which is equivalent as
    calculating K_j(x_i - x_j) where K is a Gaussian kernel.
    :param mu: ndarray, shape of [n, d], stack of mean vector of n Gaussian
               elements of d dimensions.
    :param sigma: ndarray, shape of [n, d, d], n stacked covariance matrices of
                  dimension d.
    :return: ndarray, shape of [n, n], matrix of pairwise p_i(mu_j), each entry
             [i][j] represents p_i(mu_j)
    """
    d = mu.shape[-1]
    mu_sub = mu[newaxis, ...] - mu[:, newaxis, ...]  # [n, n, d]
    numerator = squeeze(exp(-0.5 * (mu_sub @ inv(sigma))[..., newaxis, :] @ mu_sub[..., newaxis])).T
    denominator = (((2 * pi) ** d) * det(sigma)) ** 0.5
    prob_mean_mat = (numerator / denominator).T  # [i][j] --> p_i(mu_j)
    return prob_mean_mat


def prob_mean_mat_uniform(
        mu: NDArray,
        sigma: NDArray
) -> NDArray:
    """
    Calculate p_i(mu_j) matrix of pairwise elements in a mixture
    of uniform distribution. The mixture is composed of n uniform
    elements, and each entry [i][j] in the returned matrix represents
    pdf_i(mu_j), where pdf_i and mu_j are pdf of ith element and mean
    of jth element in the distribution, which is equivalent as
    calculating K_j(x_i - x_j) where K is a uniform kernel.
    :param mu: ndarray, shape of [n, d], centre value of n uniform elements
               of d dimensions.
    :param sigma: ndarray, shape of [n, d, d], stack of n diagonal matrices of size d,
                  each diagonal entry indicates the interval length of an element on
                  that dimension.
    :return: ndarray, shape of [n, n], matrix of pairwise p_i(mu_j), each entry
             [i][j] represents p_i(mu_j).
    """
    half_interval = 0.5 * diagonal(sigma, axis1=1, axis2=2)  # [n, d]
    upper = mu + half_interval  # [n, d]
    lower = mu - half_interval  # [n, d]
    p = 1. / prod(half_interval * 2, axis=1)  # [n,]
    dst_upper = mu[newaxis, ...] - upper[:, newaxis, ...]
    dst_lower = mu[newaxis, ...] - lower[:, newaxis, ...]
    prob_mean_mat = all((dst_upper <= 0) * (dst_lower >= 0), axis=-1)  # [n, n], [i][j] --> p_i contains mu_j
    prob_mean_mat = prob_mean_mat * p[:, newaxis, ...]
    return prob_mean_mat  # [n, n], [i][j] --> p_i(mu_j)


def estimate(
        c: NDArray,
        mu: NDArray,
        sigma: NDArray,
        elem: Literal["gauss", "uniform"] = 'gauss',
        **kwargs
) -> float:
    """
    Estimate entropy for a mixture distribution using the probability of
    the component means. When estimating mixture of Gaussian kernels share
    the same covariance matrix, estimation is equivalent to pairwise KL
    divergence estimation minus 2/d, thus this estimator could be used as an
    upper bound of true entropy as long as 2/d is added to the estimation,
    where d is the number of dimension.
    :param c: ndarray, shape of [n], weights of n elements, should sum up to one.
    :param mu: ndarray, shape of [n, d], indicates center of n elements of dimension d.
               If elem='gauss', mu represents means of each element;
               If elem='uniform', mu represents midpoint of supports of each element.
    :param sigma: ndarray, shape of [n, d, d], represents the "spread" of n elements of
                  dimension d.
                  If elem='gauss', sigma is n covariance matrices, thus have to be semi-definite;
                  If elem='uniform', sigma is n diagonal matrices, each entry on diagonal indicates
                  the interval on corresponding dimension, e.g., sigma[n][i][i] indicates
                  the interval of nth element on ith dimension.
    :param elem: str, optional, 'gauss' or 'uniform', indicates the element type for the
                 mixture distribution.
    :param kwargs: not used.
    :return: float, entropy estimated.
    """
    if elem == 'gauss':
        prob_mean_mat = prob_mean_mat_gauss(mu, sigma)
    elif elem == 'uniform':
        prob_mean_mat = prob_mean_mat_uniform(mu, sigma)
    else:
        raise NotImplementedError(f'Mixture of "{elem}" is not implemented.')
    # kernel_mat entries [i][j] indicates p_i(mu_j)
    ent_est = -sum(c * log(sum(c[:, newaxis, ...] * prob_mean_mat, axis=0)))
    return ent_est
