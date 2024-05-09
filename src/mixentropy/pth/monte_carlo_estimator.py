from typing import (
    Literal
)
from torch import (
    Tensor, Size, exp, zeros, pi,
    arange, int, squeeze,
)
from torch.linalg import (
    inv, det,
)
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform


def log_pdf_mv_gauss(
        c: Tensor,
        mu: Tensor,
        sigma: Tensor,
        x: Tensor
) -> Tensor:
    """
    Calculate log probabilities log(pdf(x)) of samples drown form
    mixtures of multivariate uniform distribution.
    :param c: torch tensor, shape of [b, k], weights of k elements, should
              sum up to one.
    :param mu: torch tensor, shape of [b, k, d], centre value of k uniform
               elements of d dimensions.
    :param sigma: torch tensor, shape of [b, k, d, d], k diagonal matrices of
                  size d, each diagonal entry indicates the interval length
                  of an element on that dimension.
    :param x: torch tensor, shape of [b, n, d], n samples drown from mixtures of
              d-dimensional Gaussian.
    :return: torch tensor, shape of [b, n], log pdf of n samples.
    """

    b, k, d = mu.shape
    x_mu = x[:, :, None, ...] - mu[:, None, ...]
    numerator = squeeze(exp(-0.5 * x_mu[..., None, :] @ inv(sigma)[:, None, ...] @ x_mu[..., None]))
    denominator = (((2 * pi) ** d) * det(sigma)) ** 0.5
    probs = (numerator * c[:, None, ...] / denominator[:, None, ...]).sum(dim=2)
    log_p = probs.log()
    return log_p  # [b, n]


def log_pdf_mv_uniform(
        c: Tensor,
        mu: Tensor,
        sigma: Tensor,
        x: Tensor
) -> Tensor:
    """
    Calculate log probabilities log(pdf(x)) of a batch of samples drowns from
    mixtures of multivariate Gaussian distribution.
    :param c: torch tensor, shape of [b, k], weights of k elements, should
              sum up to one batch wise.
    :param mu: torch tensor, shape of [b, k, d], centre value of k uniform
               elements of d dimensions.
    :param sigma: torch tensor, shape of [b, k, d, d], k diagonal matrices of
                  size d, each diagonal entry indicates the interval length
                  of an element on that dimension.
    :param x: torch tensor, shape of [b, n, d], batch of n samples drown from
              the mixture of d-dimensional mixtures.
    :return: torch tensor, shape of [b, n], batch of log probabilities of n samples.
    """

    half_interval = 0.5 * sigma.diagonal(dim1=2, dim2=3)  # [b, k, d]
    upper = mu + half_interval  # [b, k, d]
    lower = mu - half_interval  # [b, k, d]
    dst_upper = upper[:, None, ...] - x[:, :, None, ...]  # [b, n, k, d]
    dst_lower = x[:, :, None, ...] - lower[:, None, ...]  # [b, n, k, d]
    contains = ((dst_upper >= 0) * (dst_lower >= 0)).all(dim=3)  # [b, n, k] -> n_th data in k_th component
    probs = (1 / (half_interval * 2).prod(dim=2)[:, None, ...] * contains * c[:, None, ...]).sum(dim=2)  # [b, k]
    log_p = probs.log()
    return log_p  # [b, n]


def sample_mix_gauss(
        c: Tensor,
        mu: Tensor,
        sigma: Tensor,
        n_samples: int
) -> Tensor:
    """
    Sample datapoints from mixture of multivariate Gaussian distribution.
    It first draws n groups of samples from each component, then for each
    group, random choose one sample that drown from one of the component by
    weighted probability as the final result.
    :param c: torch tensor, shape of [b, k], weights of k elements, b is batch
              size, should sum up to one batch wise.
    :param mu: torch tensor, shape of [b, k, d], centre value of k uniform
               elements of d dimensions.
    :param sigma: torch tensor, shape of [b, k, d, d], k diagonal matrices of
                  size d, each diagonal entry indicates the interval length
                  of an element on that dimension.
    :param n_samples: int, number of datapoints to be sampled to estimate
                      expectation of log(p(x)).
    :return: torch tensor, shape of [b, n, d],batch of n samples of d dimensions
             drown from the mixture.
    """
    b, k, d = mu.shape
    raw_x = zeros((b, k, n_samples, d))
    for batch, comp in ((bat, com) for bat in range(b) for com in range(k)):
        sampler = MultivariateNormal(mu[batch][comp], sigma[batch][comp])
        raw_x[batch][comp] = sampler.sample(Size([n_samples]))
    sample_comp = zeros((b, n_samples), dtype=int)
    for batch in range(b):
        sample_comp[batch] = c[batch].multinomial(num_samples=n_samples, replacement=True)
    x = raw_x[arange(b)[..., None], sample_comp, arange(n_samples)]
    return x


def sample_mix_uniform(
        c: Tensor,
        mu: Tensor,
        sigma: Tensor,
        n_samples: int
) -> Tensor:
    """
    Sample datapoints from mixture of multivariate uniform distribution.
    It first samples n datapoints from multivariate uniform distribution
    with interval -0.5 to +0.5 on all dimensions, then scale each dimension
    to its corresponding interval, finally, shift each sample to its
    corresponding centre position.
    :param c: torch tensor, shape of [b, k], weights of k elements, should
              sum up to one batch wise.
    :param mu: torch tensor, shape of [b, k, d], centre value of k uniform elements
               of d dimensions.
    :param sigma: torch tensor, shape of [b, k, d, d], k diagonal matrices of
                  size d, each diagonal entry indicates the interval length
                  of an element on that dimension.
    :param n_samples: int, number of datapoints to be sampled to estimate expectation
                      of log(p(x)).
    :return: torch tensor, shape of [b, n, d], batch of n samples of d dimensions drown
             from the mixture.
    """

    b, k, d = mu.shape
    sampler = Uniform(-.5, .5)
    raw_x = sampler.sample(Size([b, k, n_samples, d]))
    raw_x *= sigma.diagonal(dim1=-2, dim2=-1)[..., None, :]
    raw_x += mu[..., None, :]
    sample_comp = zeros((b, n_samples), dtype=int)
    for batch in range(b):
        sample_comp[batch] = c[batch].multinomial(num_samples=n_samples, replacement=True)
    x = raw_x[arange(b)[..., None], sample_comp, arange(n_samples)]
    return x


def estimate(
        c: Tensor,
        mu: Tensor,
        sigma: Tensor,
        n_samples: int,
        elem: Literal["gauss", "uniform"] = 'gauss',
        **kwargs
) -> Tensor:
    """
    Estimate entropy for a mixture distribution based on Monte Carlo sampling,
    It first samples n datapoints from the mixture, and calculate each datapoint's
    log probability, and average the log probabilities over all datapoints to
    get the expectation of log probability.
            h(x) = -\int_x p(x) log(p(x)) dx
                 = -E_x[log(p(x))]
    As the number of samples approaches infinity, the estimated entropy becomes
    exact.
    :param c: torch tensor, shape of [b, n], weights of n elements, b is the
              batch size. Should sum up to one batch wise.

    :param mu: torch tensor, shape of [b, n, d], indicates center of n elements of dimension d.
               If elem='gauss', mu represents means of each element;
               If elem='uniform', mu represents midpoint of supports of each element.
    :param sigma: torch tensor, shape of [b, n, d, d], represents the "spread" of n elements
                  of dimension d.
                  If elem='gauss', sigma is n covariance matrices, thus have to be semi-definite;
                  If elem='uniform', sigma is n diagonal matrices, each entry on diagonal indicates
                  the interval on corresponding dimension, e.g., sigma[n][i][i] indicates
                  the interval of nth element on ith dimension.
    :param n_samples: int, number of datapoints to be sampled to estimate expectation of log(p(x)).
    :param elem: str, 'gauss' or 'uniform', indicates the element type for the mixture distribution.
    :param kwargs: not used.
    :return: torch tensor, batch of entropies estimated.
    """
    if elem == 'gauss':
        x = sample_mix_gauss(c, mu, sigma, n_samples)
        log_p = log_pdf_mv_gauss(c, mu, sigma, x)  # [b, n]
    elif elem == 'uniform':
        x = sample_mix_uniform(c, mu, sigma, n_samples)
        log_p = log_pdf_mv_uniform(c, mu, sigma, x)  # [b, n]
    else:
        raise NotImplementedError(f'Mixture of "{elem}" is not implemented.')
    return -log_p.mean()
