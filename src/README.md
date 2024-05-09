# MixEntroPY

Numpy & PyTorch vectorized implementation of various differential entropy estimators for ML & DL.

## About The Project

Information theory-inspired machine learning methods are gaining increasing interest, with estimation of the entropy as well as mutual information of random variables serving as their cornerstone.

To estimate the Shannon entropy of a _discrete_ random variable, given its probability distribution, one can simply apply the definition of Shannon entropy $H(x)=-\sum_i p(x_i)\log p(x_i)$ to obtain an accurate result.

But when it comes to estimate differential entropy, $h(x)=-\int p(x)\log p(x)dx$, from data points sampled from datasets or models, often there is no prior knowledge about the underlying distribution.

In such cases, we could make an assumption of the unknown distribution and expect the assumed distribution to have a closed-form expression for its entropy calculation, e.g., multivariate Gaussian.

_Kernel Density Estimation(KDE)_ is one of commonly used methods to approximate probability density of a distribution, However, while a single kernel may have a closed-form expression for its entropy, the mixture of these kernels typically does not.

_**This project offers several entropy estimators for these mixture distributions, (mainly focus on mixture of Gaussian and mixture of uniform), implemented with both Numpy and PyTorch.**_

Most of the estimators are **differentiable**, making them suitable for optimization purposes.

_Please see Github https://github.com/grizxlyzx/MixEntroPY for more information._
