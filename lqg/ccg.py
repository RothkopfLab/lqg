import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit


def xcorr(x, y, maxlags=60, normed=True):
    """ Compute cross correlation between two arrays along last axis of an array,
    treating other axes as batch dimensions

    Args:
        x, y: arrays of same dimensions
        maxlags: cutoff
        normed: normalize the correlations

    Returns:
        lags, correlations
    """
    Nx = x.shape[-1]

    correls = fftconvolve(x, y[..., ::-1], mode="full", axes=-1)

    if normed:
        correls = correls / np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., None]

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)

    correls = correls[..., Nx - 1 - maxlags:Nx + maxlags]
    return lags, correls


import numpy as np


def dog(x, a1, a2, mu1, mu2, sigma1, sigma2):
    g = a1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-.5 * (x - mu1) ** 2 / sigma1 ** 2)
    h = a2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-.5 * (x - mu2) ** 2 / sigma2 ** 2)

    return g - h


def skewed_gabor(x, a, mu, sigma1, sigma2, w):
    # f = np.zeros_like(x)
    f = (x >= mu) * a * np.exp(-.5 * (x - mu) ** 2 / sigma1 ** 2) * np.sin(2 * np.pi * w * (x - mu)) + (
            x < mu) * a * np.exp(-.5 * (x - mu) ** 2 / sigma2 ** 2) * np.sin(
        2 * np.pi * w * (x - mu))

    return f


def fit_dog(x, y):
    res = curve_fit(dog, x, y)
    params = res[0]
    return dict(a1=params[0], a2=params[1], mu1=params[2], mu2=params[3], sigma1=params[4], sigma2=params[5])


def fit_skewed_gabor(x, y):
    res = curve_fit(skewed_gabor, x, y, max_nfev=5000, p0=np.array([0.5, 1., 5., 2., 1.]), method="trf",
                    bounds=(np.array([0., 0., .1, .1, .1]), np.array([1., 50., 50., 50., 5.])))
    params = res[0]
    return dict(a=params[0], mu=params[1], sigma1=params[2], sigma2=params[3], w=params[4])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from lqg.io.leap import load_tracking_data

    # x = np.linspace(0, 60)

    data, _ = load_tracking_data("S1", exp=2, delay=12, clip=60, subtract_mean=False, normalize=False,
                                 path="../../data/annotated_data")

    for data_i in data:
        x = data_i.transpose((1, 0, 2))

        lags, correls = xcorr(np.diff(x[..., 1], axis=0).T, np.diff(x[..., 0], axis=0).T, maxlags=60)

        plt.plot(lags, correls.mean(axis=0))

        res = fit_skewed_gabor(lags[60:], correls.mean(axis=0)[60:])

        plt.plot(lags, skewed_gabor(lags, **res))
        plt.show()
