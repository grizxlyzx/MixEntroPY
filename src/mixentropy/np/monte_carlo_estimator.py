from typing import (
    Literal
)
from numpy.typing import (
    NDArray
)
from numpy import (
    exp, zeros, newaxis, log, pi, mean,
    squeeze, arange, diagonal, sum, all,
    prod,
)
from numpy.linalg import (
    inv, det
)
from numpy.random import (
    multivariate_normal,
    uniform, choice
)


def log_pdf_mv_gauss(
        c: NDArray,
        mu: NDArray,
        sigma: NDArray,
        x: NDArray
) -> NDArray:
    """
    Calculate log probabilities log(pdf(x)) of samples drown form a
    mixture of multivariate uniform distribution.
    :param c: ndarray, shape of [k], weights of k elements, should sum up to one.
    :param mu: ndarray, shape of [k, d], centre value of k uniform elements
               of d dimensions.
    :param sigma: ndarray, shape of [k, d, d], k diagonal matrices of size d,
                  each diagonal entry indicates the interval length of an element
                  on that dimension.
    :param x: ndarray, shape of [n, d], n samples drown from the mixture of
              d-dimensional Gaussian.
    :return: ndarray, shape of [n,], log pdf of n samples.
    """
    k, d = mu.shape
    x_mu = x[:, newaxis, ...] - mu[newaxis, ...]  # [n, k, d]
    numerator = squeeze(exp(-0.5 * x_mu[..., newaxis, :] @ inv(sigma)[newaxis, ...] @ x_mu[..., newaxis]))  # [n, k]
    denominator = (((2 * pi) ** d) * det(sigma)) ** 0.5
    probs = sum(numerator * c[newaxis, ...] / denominator[newaxis, ...], axis=1)
    log_p = log(probs)
    return log_p


def log_pdf_mv_uniform(
        c: NDArray,
        mu: NDArray,
        sigma: NDArray,
        x: NDArray
) -> NDArray:
    """
    Calculate log probabilities log(pdf(x)) of samples drown from a
    mixture of multivariate Gaussian distribution.
    :param c: ndarray, shape of [k], weights of k elements, should sum up to one.
    :param mu: ndarray, shape of [k, d], centre value of k uniform elements
               of d dimensions.
    :param sigma: ndarray, shape of [k, d, d], k diagonal matrices of size d,
                  each diagonal entry indicates the interval length of an element
                  on that dimension.
    :param x: ndarray, shape of [n, d], n samples drown from the mixture of
              d-dimensional mixture.
    :return: ndarray, shape of [n,], log probabilities of n samples.
    """
    half_interval = 0.5 * diagonal(sigma, axis1=1, axis2=2)  # [k, d]
    upper = mu + half_interval  # [k, d]
    lower = mu - half_interval  # [k, d]
    dst_upper = upper[newaxis, ...] - x[:, newaxis, ...]  # [n, k, d]
    dst_lower = x[:, newaxis, ...] - lower[newaxis, ...]  # [n, k, d]
    contains = all((dst_upper >= 0) * (dst_lower >= 0), axis=2)  # [n, k] -> n_th data in k_th component
    probs = sum((1 / prod(half_interval * 2, axis=1)) * contains * c, axis=1)  # [k,]
    log_p = log(probs)
    return log_p


def sample_mix_gauss(
        c: NDArray,
        mu: NDArray,
        sigma: NDArray,
        n_samples: int
) -> NDArray:
    """
    Sample datapoints from mixture of multivariate Gaussian distribution.
    It first draws n groups of samples from each component, then for each
    group, random choose one sample that drown from one of the component by
    weighted probability as the final result.
    :param c: ndarray, shape of [k], weights of k elements, should sum up to one.
    :param mu: ndarray, shape of [k, d], centre value of k uniform elements
               of d dimensions.
    :param sigma: ndarray, shape of [k, d, d], k diagonal matrices of size d,
                  each diagonal entry indicates the interval length of an element
                  on that dimension.
    :param n_samples: int, number of datapoints to be sampled to estimate expectation of log(p(x)).
    :return: ndarray, shape of [n, d], n samples of d dimensions drown from the mixture.
    """
    n, d = mu.shape
    raw_x = zeros((n, n_samples, d))
    for comp in range(n):
        raw_x[comp] = multivariate_normal(mu[comp], sigma[comp], size=n_samples)
    sample_comp = choice(n, size=n_samples, p=c)
    x = raw_x[sample_comp, arange(n_samples)]
    return x


def sample_mix_uniform(
        c: NDArray,
        mu: NDArray,
        sigma: NDArray,
        n_samples: int
) -> NDArray:
    """
    Sample datapoints from mixture of multivariate uniform distribution.
    It first samples n datapoints from multivariate uniform distribution
    with interval -0.5 to +0.5 on all dimensions, then scale each dimension
    to its corresponding interval, finally, shift each sample to its corresponding
    centre position.
    :param c: ndarray, shape of [k], weights of k elements, should sum up to one.
    :param mu: ndarray, shape of [k, d], centre value of k uniform elements
               of d dimensions.
    :param sigma: ndarray, shape of [k, d, d], k diagonal matrices of size d,
                  each diagonal entry indicates the interval length of an element
                  on that dimension.
    :param n_samples: int, number of datapoints to be sampled to estimate expectation of log(p(x)).
    :return: ndarray, shape of [n, d], n samples of d dimensions drown from the mixture.
    """
    n, d = mu.shape
    raw_x = uniform(-.5, .5, size=(n, n_samples, d))
    raw_x *= diagonal(sigma, axis1=1, axis2=2)[:, newaxis, :]
    raw_x += mu[:, newaxis, :]
    sample_comp = choice(n, size=n_samples, p=c)
    x = raw_x[sample_comp, arange(n_samples)]
    return x


def estimate(
        c: NDArray,
        mu: NDArray,
        sigma: NDArray,
        n_samples: int,
        elem: Literal["gauss", "uniform"] = 'gauss',
        **kwargs
) -> float:
    """
    Estimate entropy for a mixture distribution based on Monte Carlo sampling,
    It first samples n datapoints from the mixture, and calculate each datapoint's
    log probability, and average the log probabilities over all datapoints to
    get the expectation of log probability.
            h(x) = -\int_x p(x) log(p(x)) dx
                 = -E_x[log(p(x))]
    As the number of samples approaches infinity, the estimated entropy becomes
    exact.
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
    :param n_samples: int, number of datapoints to be sampled to estimate expectation of log(p(x)).
    :param elem: str, 'gauss' or 'uniform', indicates the element type for the mixture distribution.
    :param kwargs: not used.
    :return: float, entropy estimated.
    """
    if elem == 'gauss':
        x = sample_mix_gauss(c, mu, sigma, n_samples)
        log_p = log_pdf_mv_gauss(c, mu, sigma, x)  # [n,]
    elif elem == 'uniform':
        x = sample_mix_uniform(c, mu, sigma, n_samples)
        log_p = log_pdf_mv_uniform(c, mu, sigma, x)  # [n, ]
    else:
        raise NotImplementedError(f'Mixture of "{elem}" is not implemented.')

    return -mean(log_p)
