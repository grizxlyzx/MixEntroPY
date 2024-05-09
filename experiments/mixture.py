import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from scipy.stats import wishart, gamma
from mixentropy.np.monte_carlo_estimator import estimate as h_mc
from mixentropy.np.kde_estimator import estimate as h_kde
from mixentropy.np.elk_estimator import estimate as h_elk
from mixentropy.np.pairwise_dist_estimator import estimate as h_pairwise

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
LOG_BASE = 10


def mixture_single_run(x, c, mu, sigma, n_mc, elem):
    ret = np.zeros(6)
    ret[0] = x
    ret[1] = h_mc(c, mu, sigma, n_mc, elem)
    ret[2] = h_kde(c, mu, sigma, elem)
    ret[3] = h_elk(c, mu, sigma, elem)
    ret[4] = h_pairwise(c, mu, sigma, elem, 'kl')
    ret[5] = h_pairwise(c, mu, sigma, elem, 'chernoff', alpha=0.5)
    return ret


def plot(
        ret,
        title='',
        x_label='',
        y_label='',
        show=False,
        save_path=None,
        is_log_space=True
):
    x_axis = np.emath.logn(LOG_BASE, ret[0]) if is_log_space else ret[0]
    plt.plot(x_axis, ret[1], c='black', linestyle='-', label='H_MonteCarlo')
    plt.plot(x_axis, ret[2], c='purple', linestyle='-.', label='H_KDE')
    plt.plot(x_axis, ret[3], c='blue', linestyle=':', label='H_ELK')
    plt.plot(x_axis, ret[4], c='orange', linestyle=(0, (5, 1, 1, 1, 1, 1)), label='H_KL')
    plt.plot(x_axis, ret[5], c='hotpink', linestyle=(0, (3, 1, 1, 1, 1, 1, 1, 1)), label='H_BHAT')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().set_box_aspect(aspect=0.6)
    plt.legend(loc='upper left', prop={'size': 6})
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()


def mean_spread(
        k: int = 100,
        d: int = 10,
        n_mc_sample: int = 3000,
        log_start: float = -2,
        log_stop: float = 1.5,
        steps: int = 20,
        n_sweep: int = 10,
        sigma_coef: float = 1.,
        elem: Literal["gauss", "uniform"] = 'gauss'
):
    ret = np.zeros((n_sweep, 6, steps), dtype=np.float64)
    for sweep in range(n_sweep):  # sweep over n times to take average
        for i, s in enumerate(np.logspace(log_start, log_stop, num=steps, dtype=np.float64, base=LOG_BASE)):
            c = np.ones(k) / k
            mu = np.stack([np.random.multivariate_normal(np.zeros(d), s * np.eye(d)) for _ in range(k)])
            sigma = np.stack([np.eye(d)] * k) * sigma_coef
            ret[sweep, :, i] = mixture_single_run(s, c, mu, sigma, n_mc_sample, elem)
    ret = np.mean(ret, axis=0)
    return ret


def covariance_similarity(
        k=100,
        d=10,
        n_mc_sample=3000,
        log_start=0.,
        log_stop=7.,
        steps=20,
        n_sweep=10,
        elem='gauss'
):
    ret = np.zeros((n_sweep, 6, steps), dtype=np.float64)
    for sweep in range(n_sweep):
        for i, s in enumerate(np.logspace(log_start, log_stop, num=steps, dtype=np.float64, base=LOG_BASE)):
            c = np.ones(k) / k
            mu = np.stack([np.random.multivariate_normal(np.zeros(d), np.eye(d)) for _ in range(k)])
            if elem == 'gauss':
                sigma = np.stack([wishart.rvs(df=d + s, scale=1 / (d + s) * np.eye(d)) for _ in range(k)])
            elif elem == 'uniform':
                sigma = np.stack([gamma.rvs(a=1. + s, scale=1 / (1. + s)) * np.eye(d) * 2. for _ in range(k)])
            else:
                raise NotImplementedError(f'{elem} not supported.')
            ret[sweep, :, i] = mixture_single_run(s, c, mu, sigma, n_mc_sample, elem)
    ret = np.mean(ret, axis=0)
    return ret


def cluster_spread(
        k=100,
        d=10,
        n_group=5,
        n_mc_sample=3000,
        log_start=-2.5,
        log_stop=0.5,
        steps=20,
        n_sweep=20,
        sigma_coef=1.,
        elem='gauss'
):
    rng = np.random.default_rng()
    ret = np.zeros((n_sweep, 6, steps), dtype=np.float64)
    for sweep in range(n_sweep):
        for i, s in enumerate(np.logspace(log_start, log_stop, num=steps, dtype=np.float64, base=LOG_BASE)):
            c = np.ones(k) / k
            mu_cluster = np.stack([np.random.multivariate_normal(np.zeros(d), s * np.eye(d)) for _ in range(n_group)])
            mu = rng.choice(a=mu_cluster, size=k, axis=0)
            sigma = np.stack([np.eye(d) * sigma_coef] * k)
            ret[sweep, :, i] = mixture_single_run(s, c, mu, sigma, n_mc_sample, elem)
    ret = np.mean(ret, axis=0)
    return ret


def dimension(
        k=100,
        n_mc_sample=3000,
        d_min=1,
        d_max=26,
        n_sweep=5,
        sigma_coef=1.,
        elem='gauss'
):
    ret = np.zeros((n_sweep, 6, d_max - d_min), dtype=np.float64)
    for sweep in range(n_sweep):
        for i, d in enumerate(range(d_min, d_max)):
            c = np.ones(k) / k
            mu = np.stack([np.random.multivariate_normal(np.zeros(d), np.eye(d)) for _ in range(k)])
            sigma = np.stack([np.eye(d) * sigma_coef] * k)
            ret[sweep, :, i] = mixture_single_run(d, c, mu, sigma, n_mc_sample, elem)
    ret = np.mean(ret, axis=0)
    return ret


def run_gauss(step=20, n_sweep=1):
    ret = mean_spread(
        k=100,
        d=10,
        n_mc_sample=3000,
        log_start=-2.5,
        log_stop=1.5,
        steps=step,
        n_sweep=n_sweep,
        sigma_coef=1.,
        elem='gauss'
    )
    plot(
        ret,
        title='Mean Spread (Gaussian)',
        x_label=r'$\log_{10}$(Mean Spread)',
        y_label='Entropy(nats)',
        show=False,
        save_path=os.path.join(SAVE_DIR, 'GaussianMeanSpread.png')
    )
    print('Mean Spread (Gaussian) == SAVED ==')
    ret = covariance_similarity(
        k=100,
        d=10,
        n_mc_sample=3000,
        log_start=0,
        log_stop=7,
        steps=step,
        n_sweep=n_sweep,
        elem='gauss'
    )
    plot(
        ret,
        title='Covariance Similarity (Gaussian)',
        x_label=r'$\log_{10}$(Covariance Similarity)',
        y_label='Entropy(nats)',
        show=False,
        save_path=os.path.join(SAVE_DIR, 'GaussianCovarianceSimilarity.png')
    )
    print('Covariance Similarity (Gaussian) == SAVED ==')
    ret = cluster_spread(
        k=100,
        d=10,
        n_group=5,
        n_mc_sample=3000,
        log_start=-2.5,
        log_stop=0.5,
        steps=step,
        n_sweep=n_sweep,
        sigma_coef=1.,
        elem='gauss'
    )
    plot(
        ret,
        title='Cluster Spread (Gaussian)',
        x_label=r'$\log_{10}$(Cluster Spread)',
        y_label='Entropy(nats)',
        show=False,
        save_path=os.path.join(SAVE_DIR, 'GaussianClusterSpread.png')
    )
    print('Cluster Spread (Gaussian) == SAVED ==')
    ret = dimension(
        k=100,
        n_mc_sample=3000,
        d_min=1,
        d_max=25,
        n_sweep=n_sweep,
        sigma_coef=1.,
        elem='gauss'
    )
    plot(
        ret,
        title='Dimension (Gaussian)',
        x_label='Dimension',
        y_label='Entropy(nats)',
        show=False,
        save_path=os.path.join(SAVE_DIR, 'GaussianDimension.png'),
        is_log_space=False
    )
    print('Dimension (Gaussian) == SAVED ==')


def run_uniform(step=20, n_sweep=1):
    ret = mean_spread(
        k=100,
        d=10,
        n_mc_sample=3000,
        log_start=-4,
        log_stop=2,
        steps=step,
        n_sweep=n_sweep,
        sigma_coef=2.,
        elem='uniform'
    )
    plot(
        ret,
        title='Mean Spread (Uniform)',
        x_label=r'$\log_{10}$(Mean Spread)',
        y_label='Entropy(nats)',
        show=False,
        save_path=os.path.join(SAVE_DIR, 'UniformMeanSpread.png')
    )
    print('Mean Spread (Uniform) == SAVED ==')
    ret = covariance_similarity(
        k=100,
        d=10,
        n_mc_sample=3000,
        log_start=-3.5,
        log_stop=3,
        steps=step,
        n_sweep=n_sweep,
        elem='uniform'
    )
    plot(
        ret,
        title='Width Variability (Uniform)',
        x_label=r'$\log_{10}$(Width Variability)',
        y_label='Entropy(nats)',
        show=False,
        save_path=os.path.join(SAVE_DIR, 'UniformWidthVariability.png')
    )
    print('Width Variability (Uniform) == SAVED ==')
    ret = cluster_spread(
        k=100,
        d=10,
        n_group=5,
        n_mc_sample=3000,
        log_start=-6,
        log_stop=0,
        steps=step,
        n_sweep=n_sweep,
        sigma_coef=2.,
        elem='uniform'
    )
    plot(
        ret,
        title='Cluster Spread (Uniform)',
        x_label=r'$\log_{10}$(Cluster Spread)',
        y_label='Entropy(nats)',
        show=False,
        save_path=os.path.join(SAVE_DIR, 'UniformClusterSpread.png')
    )
    print('Cluster Spread (Uniform) == SAVED ==')
    ret = dimension(
        k=100,
        n_mc_sample=3000,
        d_min=1,
        d_max=16,
        n_sweep=n_sweep,
        sigma_coef=2.,
        elem='uniform'
    )
    plot(
        ret,
        title='Dimension (Uniform)',
        x_label='Dimension',
        y_label='Entropy(nats)',
        show=False,
        save_path=os.path.join(SAVE_DIR, 'UniformDimension.png'),
        is_log_space=False
    )
    print('Dimension (Uniform) == SAVED ==')


if __name__ == '__main__':
    run_gauss(step=50, n_sweep=50)
    run_uniform(step=50, n_sweep=50)
