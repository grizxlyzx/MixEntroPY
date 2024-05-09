from numpy.typing import (
    NDArray
)
from numpy import (
    eye, transpose, sum
)
from numpy.linalg import (
    slogdet
)


def logdet_entropy(
        x: NDArray,
        beta: float = 1
) -> float:
    r"""
    Estimate differential entropy for a multivariate Gaussian (or any other)
    distribution by calculating the logarithm of determinant of covariance
    matrix:

            H(X) = 1/2 * logdet(I + beta * cov)

    Covariance matrix is calculated by:

            cov(x) = X.T @ X / n

    :param x: ndarray, shape of [n, d], n sampled datapoint, each point has d dimensions
    :param beta: float, scale factor of covariance matrix to balance the bias introduced
                 by added Gaussian noise.
    :return: float, estimated different entropy.
    """
    n, d = x.shape
    x = x - x.mean(axis=0)
    _, logdet = slogdet(eye(d) + (beta * (x.T @ x) / n))
    ent = 0.5 * logdet
    return ent


def logdet_joint_entropy(
        x: NDArray,
        beta: float
) -> float:
    r"""
    Estimate differential joint entropy for a set of multivariate Gaussian (or any other)
    distribution, by calculating the logarithm of determinant of pairwise datapoint inner
    product of each distribution:

            H(X1, X2, ..., Xn) = 1/2 * logdet(I + beta * sum(K))

        where K is the pairwise similarity kernel between each pair of samples:

            Kn = Xn @ Xn.T

    :param x: ndarray, shape of [k, n, d], k distributions, each has n sampled datapoints,
              each datapoint has d dimensions.
    :param beta: float, scale factor of covariance matrix to balance the bias introduced
                 by added Gaussian noise.
    :return: float, estimated differential joint entropy.
    """
    _, n, _ = x.shape
    x = x - x.mean(axis=1, keepdims=True)
    kernel = x @ transpose(x, axes=(0, 2, 1))  # [k, n, d] @ [k, d, n] -> [k, n, n]
    _, logdet = slogdet(eye(n) + beta * sum(kernel, axis=0) / n)
    ent = 0.5 * logdet
    return ent
