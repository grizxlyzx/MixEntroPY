from typing import (
    Literal
)

from numpy.typing import (
    NDArray
)
from numpy import (
    pi, sum, log, trace, newaxis,
    exp, inf, squeeze, diagonal,
    maximum, minimum, all, errstate
)
from numpy.linalg import (
    inv, slogdet
)


def div_mat_kl_gaussian(
        mu: NDArray,
        sigma: NDArray,
        d: int,
        **kwargs
) -> NDArray:
    """
    Closed form KL divergence matrix of pairwise elements in a mixture of
    gaussian distribution.
    :param mu: ndarray, shape of [n, d], stack of mean vector of n Gaussian
               elements of d dimensions.
    :param sigma: ndarray, shape of [n, d, d], n stacked covariance matrices of
                  dimension d.
    :param d: int, number of dimensions.
    :return: ndarray, shape of [n, n], matrix of pairwise KL divergence.
    """

    inv_sigma = inv(sigma)  # [n, d, d]
    _, ld_sigma = slogdet(sigma)  # [n]
    mu_sub = mu[:, newaxis, ...] - mu[newaxis, ...]  # mu2 - mu1, [n, n, d]
    div_mat = inv_sigma[newaxis, ...] @ sigma[:, newaxis, ...]
    div_mat = trace(div_mat, axis1=-2, axis2=-1)  # [n, n]
    div_mat += -d
    div_mat += squeeze((mu_sub @ inv_sigma)[..., newaxis, :] @ mu_sub[..., newaxis]).T
    div_mat += ld_sigma[newaxis, ...] - ld_sigma[:, newaxis, ...]
    div_mat *= 0.5
    return div_mat


def div_mat_kl_uniform(
        mu: NDArray,
        sigma: NDArray,
        **kwargs
) -> NDArray:
    """
    Closed form KL divergence matrix of pairwise elements in a mixture of
    uniform distribution.
    :param mu: ndarray, shape of [n, d], centre value of n uniform elements
               of d dimensions.
    :param sigma: ndarray, shape of [n, d, d], n diagonal matrices of size d,
                  each diagonal entry indicates the interval length of an element
                  on that dimension.
    :param kwargs: not used.
    :return: ndarray, shape of [n, n], matrix of pairwise KL divergence.
    """
    # if two uniform element have some non-overlap region,
    # KL divergence between them is positive infinity.
    # check if non-overlapping region exist
    half_interval = 0.5 * diagonal(sigma, axis1=1, axis2=2)  # [n, d]
    log_volume = sum(log(half_interval * 2), axis=1)
    upper = mu + half_interval  # [n, d]
    lower = mu - half_interval  # [n, d]
    dst_upper = upper[newaxis, ...] - upper[:, newaxis, ...]  # [n, n, d]
    dst_lower = lower[newaxis, ...] - lower[:, newaxis, ...]  # [n, n, d]
    contains = all((dst_upper <= 0) * (dst_lower >= 0), axis=-1)  # [n, n], [i][j] --> p_i contains p_j
    div_mat = (log_volume[:, newaxis, ...] - log_volume[newaxis, ...]) * contains
    div_mat[~contains] = inf
    return div_mat


def div_mat_chernoff_alpha_gaussian(
        mu: NDArray,
        sigma: NDArray,
        alpha: float = 0.5,
        **kwargs
) -> NDArray:
    """
    Closed form Chernoff-alpha distance matrix of pairwise elements
    in a mixture of gaussian distribution.
    Chernoff-alpha distance for alpha=0.5(as default) is known
    as Bhattacharyya distance.
    :param mu: ndarray, shape of [n, d], stack of mean vector of n Gaussian
               elements of d dimensions.
    :param sigma: ndarray, shape of [n, d, d], n stacked covariance matrices of
                  dimension d.
    :param alpha: float, Chernoff α-coefficient, must be in [0, 1], otherwise the
                  equation is not a valid distance function.
    :param kwargs: not used.
    :return: ndarray, shape of [n, n], matrix of pairwise Chernoff-alpha distance
    """
    mu_sub = mu[:, newaxis, ...] - mu[newaxis, ...]
    weighted_sigma = ((1 - alpha) * sigma)[newaxis, ...] + (alpha * sigma)[:, newaxis, ...]
    _, ld_sigma = slogdet(sigma)
    _, ld_weighted_sigma = slogdet(weighted_sigma)
    div_mat = squeeze((mu_sub[..., newaxis, :] @ inv(weighted_sigma)) @ mu_sub[..., newaxis])
    div_mat *= ((1 - alpha) * alpha)
    div_mat += ld_weighted_sigma - (((1 - alpha) * ld_sigma)[..., newaxis] + (alpha * ld_sigma)[..., newaxis, :])
    div_mat *= 0.5
    return div_mat


def div_mat_chernoff_alpha_uniform(
        mu: NDArray,
        sigma: NDArray,
        alpha: float = 0.5,
        **kwargs
) -> NDArray:
    """
    Closed form Chernoff-alpha distance matrix of pairwise elements
    in a mixture of uniform distribution.
    Chernoff-alpha distance for alpha=0.5(as default) is known
    as Bhattacharyya distance.
    :param mu: ndarray, shape of [n, d], centre value of n uniform elements
               of d dimensions.
    :param sigma: ndarray, shape of [n, d, d], n diagonal matrices of size d,
                  each diagonal entry indicates the interval length of an element
                  on that dimension.
    :param alpha: float, Chernoff α-coefficient, must be in [0, 1], otherwise the
                  equation is not a valid distance function.
    :return: ndarray, shape of [n, n], matrix of pairwise Chernoff-alpha distance.
    """
    half_interval = 0.5 * diagonal(sigma, axis1=1, axis2=2)  # [n, d]
    log_volume = sum(log(half_interval * 2), axis=1)  # [n,]
    upper = mu + half_interval  # [n, d]
    lower = mu - half_interval  # [n, d]
    overlap_lower = maximum(lower[newaxis, ...], lower[:, newaxis, ...])  # [n, n, d]
    overlap_upper = minimum(upper[newaxis, ...], upper[:, newaxis, ...])  # [n, n, d]
    # get rid of "divide by zero in log" warning, since we will encounter log(0)
    # and get -inf as expected
    with errstate(divide="ignore"):
        log_overlap_volume = sum(log(maximum(overlap_upper - overlap_lower, 0)), axis=2)  # [n, n]
    div_mat = alpha * log_volume[newaxis, ...] + (1 - alpha) * log_volume[:, newaxis, ...] - log_overlap_volume
    return div_mat


def estimate(
        c: NDArray,
        mu: NDArray,
        sigma: NDArray,
        elem: Literal["gauss", "uniform"] = 'gauss',
        dist: Literal["kl", "chernoff"] = 'kl',
        **kwargs
) -> float:
    """
    Estimate entropy for a mixture distribution based on pairwise distances b/w elements,
    Elements of the mixture could be "gauss" for Gaussian or "uniform" for uniform.
    Distance function could be "kl" for Kullback–Leibler divergence, or "chernoff"
    for Chernoff-alpha-distance.
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
    :param elem: str, 'gauss' or 'uniform', indicates the element type for the mixture distribution.
    :param dist: str, 'kl' or 'chernoff', indicates the distance function for measuring pairwise distances
                 between each element.
                 When using 'kl', estimation will be an upper bound on the true entropy;
                 When using 'chernoff', estimation will be a lower bound on the true entropy.
    :param kwargs: extra arguments for pairwise distance calculation.
    :return: float, entropy estimated.
    """
    if elem == 'gauss':
        n, d = mu.shape
        ent_x = 0.5 * (slogdet(sigma)[1] + d * log(2 * pi) + d)  # calculate individual entropies
        if dist == 'kl':
            div_mat = div_mat_kl_gaussian(mu, sigma, d, **kwargs)
        elif dist == 'chernoff':
            div_mat = div_mat_chernoff_alpha_gaussian(mu, sigma, **kwargs)
        else:
            raise NotImplementedError(f'Distance function: "{dist}" is not implemented')

    elif elem == 'uniform':
        ent_x = sum(log(diagonal(sigma, axis1=1, axis2=2)), axis=1)  # calculate individual entropies
        if dist == 'kl':
            div_mat = div_mat_kl_uniform(mu, sigma, **kwargs)
        elif dist == 'chernoff':
            div_mat = div_mat_chernoff_alpha_uniform(mu, sigma, **kwargs)
        else:
            raise NotImplementedError(f'Distance function: "{dist}" is not implemented')

    else:
        raise NotImplementedError(f'Mixture of "{elem}" is not implemented.')

    ent_xc = sum(c * ent_x)  # avg. over elements entropy
    ent_est = c[newaxis, ...] * exp(-div_mat)
    ent_est = c * log(sum(ent_est, axis=1))
    ent_est = ent_xc - sum(ent_est)
    return ent_est
