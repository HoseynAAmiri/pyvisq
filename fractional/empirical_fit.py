from scipy.optimize import curve_fit
from sklearn.metrics import r2_score as r2
import numpy as np
import plotly.graph_objs as go
import os


def func(t, A, alpha, tau, B):
    return A * t ** (-alpha) * np.exp(- tau * t) + B


def func_fit(x, y):
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    x = (x - x_min) / x_max
    y = (y - y_min) / y_max
    x = x[1:]
    y = y[1:]
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
    popt, _ = curve_fit(func, x, y, bounds=bounds)
    y_pred = func(x, *popt)
    r2_score = r2(y, y_pred)
    return popt, r2_score, y_pred * y_max + y_min


fig = go.Figure()

idx = 19
exp_name = 'v2'
file_path = os.path.join('./data', exp_name)
file_name = os.path.join(file_path, 'dwell_key' + str(idx) + '.npz')
data = np.load(file_name)
t = data['t']
f = data['f']
fig.add_trace(go.Scattergl(
    x=t, y=f, mode='markers', marker={'color': 'blue', 'size': 6}, name='data'))

_, r2_score, y_pred = func_fit(t, f)
print(r2_score)
fig.add_trace(go.Scattergl(
    x=t[1:], y=y_pred, mode='markers', marker={'color': 'red', 'size': 5}, name='fit'))
fig.update_layout(
    xaxis_type="log",
    yaxis_type="log"
)
fig.show()
