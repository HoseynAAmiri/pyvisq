import numpy as np
from scipy.optimize import curve_fit
from functools import partial


# Functions
def powerlaw(x, a, b):
    return a * (x ** b)


def exponent(t, y_0, tau, y_f):
    return (y_0 - y_f) * np.exp(-(t-t[0]) * tau) + y_f


def biexponent(t, y_0, c, tau1, tau2, y_f):
    return (y_0 - y_f) * (c * np.exp(-(t-t[0]) * tau1) + (1 - c) * np.exp(-(t-t[0]) * tau2)) + y_f


def poroelastic(t, y_0, c, tau1, tau2, y_f):
    return (y_0 - y_f) * (c * np.exp(-np.sqrt(t-t[0]) * tau1) + (1 - c) * np.exp(-(t-t[0]) * tau2)) + y_f


def hertzian(ind, diameter, e_star):
    # ind and diameter units are um and the force unit is nN
    return (4 / 3) * e_star * ((0.5 * diameter) ** 0.5) * (ind ** 1.5) * 1e-3


# Fitting
def lin_fit(x, y):
    return np.polyfit(x, y, 1)[0]  # returns the slope


def powerlaw_fit(x, y):
    bounds = ([0, 0], [np.inf, np.inf])
    return curve_fit(powerlaw, x, y, bounds=bounds)


def exponent_fit(x, y, bound=False):
    if bound:
        def wrapper(t, tau):
            return exponent(t, y[0], tau, y[-1])

        bounds = ([0], [np.inf])
        return curve_fit(wrapper, x, y, bounds=bounds)
    else:
        def wrapper(t, tau, y_f):
            return exponent(t, y[0], tau, y_f)

        bounds = ([0, -np.inf], [np.inf, np.inf])
        return curve_fit(wrapper, x, y, bounds=bounds)


def biexponent_fit(x, y, bound=False):
    if bound:
        def wrapper(t, c, tau1, tau2):
            return biexponent(t, y[0], c, tau1, tau2, y[-1])

        bounds = ([0, 0, 0], [1, np.inf, np.inf])
        return curve_fit(wrapper, x, y, bounds=bounds)
    else:
        def wrapper(t, c, tau1, tau2, y_f):
            return biexponent(t, y[0], c, tau1, tau2, y_f)

        bounds = ([0, 0, 0, -np.inf], [1, np.inf, np.inf, np.inf])
        return curve_fit(wrapper, x, y, bounds=bounds)


def poroelastic_fit(x, y, bound=False):
    if bound:
        def wrapper(t, c, tau1, tau2):
            return poroelastic(t, y[0], c, tau1, tau2, y[-1])

        bounds = ([0, 0, 0], [1, np.inf, np.inf])
        return curve_fit(wrapper, x, y, bounds=bounds)
    else:
        def wrapper(t, c, tau1, tau2, y_f):
            return poroelastic(t, y[0], c, tau1, tau2, y_f)

        bounds = ([0, 0, 0, -np.inf], [1, np.inf, np.inf, np.inf])
        return curve_fit(wrapper, x, y, bounds=bounds)


def hertzian_fit(x, y, diameter=5):
    def wrapper(ind, e_star):
        return hertzian(ind, diameter, e_star)

    return curve_fit(wrapper, x, y)


if __name__ == '__main__':
    pass
