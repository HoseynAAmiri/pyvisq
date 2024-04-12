import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from fit import *


def get_approach(data):
    t, eps, sig = data
    max_index = np.argmax(sig)
    return t[:max_index+1], eps[:max_index+1], sig[:max_index+1]


def get_dwell(data):
    t, eps, sig = data
    max_index = np.argmax(sig)
    return t[max_index:], eps[max_index:], sig[max_index:]


I = 0.5
D = 1
E1 = 20
E2 = 30
T = 2
S = 5

DT = np.array([2, 1, 0, -1, -2])
cmap = plt.get_cmap('Blues')
norm = Normalize(vmin=0, vmax=len(DT)+3)
sm = ScalarMappable(norm=norm, cmap=cmap)

t_ind = []
data = []
plt.figure(figsize=(5, 4))
for i, dt in enumerate(DT):
    I = 0.5
    D = 1
    D = (10.0**dt) * T
    ID = I / D
    t = np.linspace(0, 2 * D, 1001)

    epsilon_t = (I * t / D) * (1 - np.heaviside(t - D, 0.5)) + \
        I * np.heaviside(t - D, 0.5)
    sigma_t = I*(-E1*T*(-np.exp(D/T) + np.exp(t/T))*np.heaviside(-D + t, 0.5) + E1*t*np.exp(t/T) - E1*(-D + T*np.exp((D - t)/T) - T + t)*np.exp(t/T)
                 * np.heaviside(-D + t, 0.5) - E2*T*(-np.exp(D/T) + np.exp(t/T))*np.heaviside(-D + t, 0.5) + E2*T*np.exp(t/T) - E2*T)*np.exp(-t/T)/D

    index_S = np.where(sigma_t >= S)[0][0]
    # print(index_S, t[index_S], epsilon_t[index_S], sigma_t[index_S])
    D = t[index_S]
    t = np.linspace(0, 2 * D, 2001)
    if 2 * D < 100:
        t = np.hstack((t, np.linspace(2 * D, 100, 1001)))
    I = ID * D
    epsilon_t = (I * t / D) * (1 - np.heaviside(t - D, 0.5)) + \
        I * np.heaviside(t - D, 0.5)
    sigma_t = I*(-E1*T*(-np.exp(D/T) + np.exp(t/T))*np.heaviside(-D + t, 0.5) + E1*t*np.exp(t/T) - E1*(-D + T*np.exp((D - t)/T) - T + t)*np.exp(t/T)
                 * np.heaviside(-D + t, 0.5) - E2*T*(-np.exp(D/T) + np.exp(t/T))*np.heaviside(-D + t, 0.5) + E2*T*np.exp(t/T) - E2*T)*np.exp(-t/T)/D

    if int(D) == D:
        D = int(D)

    if i == 0:
        eps = epsilon_t

    t1 = t[t <= D] / D
    t2 = 1 + (t[t > D] - D) / (t[-1] - D)
    tp = np.hstack((t1, t2))
    sigma_p = sigma_t[t > D]
    sigma_p = (sigma_p - sigma_p[-1]) / (sigma_p[0] - sigma_p[-1])

    # plt.plot(t[t > D] - D, sigma_t[t > D], color=sm.to_rgba(
    #     i+3), label="$t_{ind}$=" + str(np.round(D, 5)))
    l = T / D
    if l < 10:
        l = np.round(l, 2)
    else:
        l = int(l)

    t_ind.append(l)
    plt.plot(epsilon_t, sigma_t, color=sm.to_rgba(
        i+3), label="$\\tau/t_{ind}$=" + str(l))

    # plt.show()
    data.append([t, epsilon_t, sigma_t])

plt.plot(epsilon_t, epsilon_t * (E1 + E2), '--g')
plt.plot(eps, eps * E1, '--g')

ax1 = plt.gca()
ax1.set_xlabel('$\\epsilon$')

color1 = 'tab:blue'
ax1.set_ylabel('$\\sigma$', color=color1)
ax1.tick_params(axis='y', labelcolor=color1, colors=color1)
ax1.spines['left'].set_color(color1)
plt.legend()

# plt.twinx()
# ax2 = plt.gca()
# color2 = 'tab:red'
# plt.plot(t, epsilon_t, 'r', label='Strain')
# ax2.set_ylabel('$\\epsilon$', color=color2)
# ax2.tick_params(axis='y', labelcolor=color2, colors=color2)
# ax2.spines['right'].set_color(color2)

plt.show()

tau = []
idx = []
for i, d in enumerate(data):
    t_appr, eps_appr, sig_appr = get_approach(d)

    t_dwell, eps_dwell, sig_dwell = get_dwell(d)
    popt, _ = exponent_fit(t_dwell, sig_dwell, bound=False)

    tau.append(1/popt[0])
    idx.append(i)
    plt.scatter(t_dwell - t_dwell[0], sig_dwell, color=sm.to_rgba(
        i+3), label="$\\tau/t_{ind}$=" + str(t_ind[i]), s=10)
    plt.plot(t_dwell - t_dwell[0],
             exponent(t_dwell - t_dwell[0], sig_dwell[0], *popt), 'r')

    print(sig_dwell[-1], eps_dwell[0], sig_dwell[0] - sig_dwell[-1])
    # print((sig_dwell[0] - sig_dwell[-1]) / E2)

plt.ylabel('$\\sigma$')
plt.xlabel('Time')
plt.legend()
plt.show()
'''
print(tau)
plt.scatter(idx, tau)
plt.show()


tau_f = []
tau_s = []
i = 0
idx = []
for d in data:
    t_appr, eps_appr, sig_appr = get_approach(d)

    t_dwell, eps_dwell, sig_dwell = get_dwell(d)
    popt, _ = biexponent_fit(t_dwell, sig_dwell, bound=False)
    tau_f.append(max(popt[1], popt[2]))
    tau_s.append(min(popt[1], popt[2]))
    idx.append(i)
    i += 1
    plt.scatter(t_dwell - t_dwell[0], sig_dwell, c='b')
    plt.plot(t_dwell - t_dwell[0],
             biexponent(t_dwell - t_dwell[0], sig_dwell[0], *popt), 'r')
    plt.show()

print(tau_f)
print(tau_s)
plt.scatter(tau_f, tau_s, c=idx)
plt.colorbar()
plt.show()
'''
