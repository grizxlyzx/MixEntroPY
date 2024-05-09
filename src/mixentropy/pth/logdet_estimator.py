from torch import (
    Tensor, logdet, eye
)


def logdet_entropy(
        x: Tensor,
        beta: float = 1
) -> Tensor:
    r"""
    Estimate differential entropy for a multivariate Gaussian (or any other)
    distribution by calculating the logarithm of determinant of covariance
    matrix:

            H(X) = 1/2 * logdet(I + beta * cov)

    Covariance matrix is calculated by:

            cov(x) = X.T @ X / n

    :param x: torch tensor, shape of [b, n, d], n sampled datapoint, each point has d
              dimensions, b is batch dimension.
    :param beta: float, scale factor of covariance matrix to balance the bias introduced
                 by added Gaussian noise.
    :return: torch tensor, shape of [b,], batch of estimated different entropies.
    """
    b, n, d = x.shape
    x = x - x.mean(dim=1, keepdim=True)
    cov = (x.transpose(dim0=1, dim1=2) @ x) / n
    i = eye(d).repeat(b, 1, 1)
    ent = 0.5 * logdet(i + (beta * cov))
    return ent


def logdet_joint_entropy(
        x: Tensor,
        beta: float = 1
) -> Tensor:
    r"""
    Estimate differential joint entropy for a set of multivariate Gaussian (or any other)
    distribution, by calculating the logarithm of determinant of pairwise datapoint inner
    product of each distribution:

            H(X1, X2, ..., Xn) = 1/2 * logdet(I + beta * sum(K))

        where K is the pairwise similarity kernel between each pair of samples:

            Kn = Xn @ Xn.T

    :param x: torch tensor, shape of [b, k, n, d], k distributions, each has n sampled
              datapoints, each datapoint has d dimensions, b is the batch dimension.
    :param beta: float, scale factor of covariance matrix to balance the bias introduced
                 by added Gaussian noise.
    :return: torch tensor, shape of [b,], batch of estimated differential joint entropies.
    """
    b, _, n, _ = x.shape  # [batch, n_dist, n_data, n_dim]
    x = x - x.mean(dim=2, keepdim=True)
    i = eye(n).repeat(b, 1, 1)  # [b, d, d]
    kernel = x @ x.transpose(dim0=2, dim1=3)  # [b, k, n, d] @ [b, k, d, n] -> [b, k, n, n]
    ent = 0.5 * logdet(i + (beta * kernel.sum(dim=1)) / n)
    return ent
