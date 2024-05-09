from typing import (
    Literal
)
from numpy.typing import (
    NDArray
)
from numpy import (
    newaxis, pi, exp, sum, log,
    squeeze, diagonal, maximum,
    minimum, errstate
)
from numpy.linalg import (
    inv, det
)


def elk_mat_gauss(
        mu: NDArray,
        sigma: NDArray
) -> NDArray:
    r"""
    Calculate Expected Likelihood Kernel (ELK) matrix of pairwise elements in a mixture
    of Gaussian distribution. The mixture is composed of n Gaussian elements, and
    each entry [i][j] in the returned matrix represents ELK(p_i, p_j), where p_i and
    p_j are ith and jth elements in the distribution.
    For Gaussian elements:
            ELK(p_i, p_j) = q_ji(mu_i)
    where:
            q_ji = N(mu_j, sigma_i + sigma_j)
    mu_* and sigma_* are mean vector and covariance matrix of an element respectively.

    :param mu: ndarray, shape of [n, d], stack of mean vector of n Gaussian
               elements of d dimensions.
    :param sigma: ndarray, shape of [n, d, d], n stacked covariance matrices of
                  dimension d.
    :return: ndarray, shape of [n, n], symmetric matrix of pairwise expected likelihood
             kernel.
    """

    d = mu.shape[-1]
    sigma_sum = sigma[newaxis, ...] + sigma[:, newaxis, ...]  # [n, n, d, d], symmetric
    mu_sub = mu[newaxis, ...] - mu[:, newaxis, ...]  # [n, n, d]
    numerator = squeeze(exp(-0.5 * (mu_sub[..., newaxis, :] @ inv(sigma_sum)) @ mu_sub[..., newaxis]))  # [n, n]
    denominator = (((2 * pi) ** d) * det(sigma_sum)) ** 0.5  # [n, n]
    elk_mat = numerator / denominator  # [n, n]
    return elk_mat


def elk_mat_uniform(
        mu: NDArray,
        sigma: NDArray
) -> NDArray:
    r"""
    Calculate Expected Likelihood Kernel (ELK) matrix of pairwise elements in a mixture
    of uniform distribution. The mixture is composed of n uniform elements, and
    each entry [i][j] in the returned matrix represents ELK(p_i, p_j), where p_i and
    p_j are ith and jth elements in the distribution.
    For uniform elements,
            ELK(p_i, p_j) = (V_i∩j) / (V_i * V_j)
    where V_* is the "volume" of hyper rectangle of that element.

    :param mu: ndarray, shape of [n, d], centre value of n uniform elements
               of d dimensions.
    :param sigma: ndarray, shape of [n, d, d], stack of n diagonal matrices of size d,
                  each diagonal entry indicates the interval length of an element on
                  that dimension.
    :return: ndarray, shape of [n, n], symmetric matrix of pairwise expected likelihood
             kernel.
    """

    half_interval = 0.5 * diagonal(sigma, axis1=1, axis2=2)  # [n, d]
    log_volume = sum(log(half_interval * 2), axis=1)  # [n,]
    log_volume_prod = log_volume[newaxis, ...] + log_volume[:, newaxis, ...]  # [n, n]
    upper = mu + half_interval  # [n, d]
    lower = mu - half_interval  # [n, d]
    overlap_lower = maximum(lower[newaxis, ...], lower[:, newaxis, ...])  # [n, n, d]
    overlap_upper = minimum(upper[newaxis, ...], upper[:, newaxis, ...])  # [n, n, d]
    # get rid of "divide by zero in log" warning, since we will encounter log(0)
    # and get -inf as expected
    with errstate(divide="ignore"):
        log_overlap_volume = sum(log(maximum(overlap_upper - overlap_lower, 0)), axis=2)  # [n, n]
    elk_mat = exp(log_overlap_volume - log_volume_prod)
    return elk_mat


def estimate(
        c: NDArray,
        mu: NDArray,
        sigma: NDArray,
        elem: Literal["gauss", "uniform"] = 'gauss',
        **kwargs
) -> float:
    r"""
    Estimate entropy for a mixture distribution based on Expected Likelihood
    Kernel (ELK). Estimation is a lower bound of true entropy.
    Elements of the mixture could be "gauss" for Gaussian or "uniform" for uniform.

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
        elk_mat = elk_mat_gauss(mu, sigma)
    elif elem == 'uniform':
        elk_mat = elk_mat_uniform(mu, sigma)
    else:
        raise NotImplementedError(f'Mixture of "{elem}" is not implemented.')
    ent_est = -sum(c * log(sum(c[:, newaxis, ...] * elk_mat, axis=0)))
    return ent_est
