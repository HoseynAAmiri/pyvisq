import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


I = 0.5
D = 1
E1 = 20
E2 = 30
T = 10

TD = np.array([-2, -1, 0, 1, 2])
cmap = plt.get_cmap('Blues')
norm = Normalize(vmin=min(TD), vmax=max(TD))
sm = ScalarMappable(norm=norm, cmap=cmap)

plt.figure(figsize=(5, 4))
for i, td in enumerate(TD):
    T = (10.0**td) * D
    t = np.linspace(0, 2, 200)
    epsilon_t = (I * t / D) * (1 - np.heaviside(t - D, 0.5)) + \
        I * np.heaviside(t - D, 0.5)
    sigma_t = I*(-E1*T*(-np.exp(D/T) + np.exp(t/T))*np.heaviside(-D + t, 0.5) + E1*t*np.exp(t/T) - E1*(-D + T*np.exp((D - t)/T) - T + t)*np.exp(t/T)
                 * np.heaviside(-D + t, 0.5) - E2*T*(-np.exp(D/T) + np.exp(t/T))*np.heaviside(-D + t, 0.5) + E2*T*np.exp(t/T) - E2*T)*np.exp(-t/T)/D

    if int(T) == T:
        T = int(T)

    if i == 0:
        plt.plot(t, epsilon_t * E1, '--g')

    plt.plot(t, sigma_t, color=sm.to_rgba(T), label="$\\tau$=" + str(T))

    if i == len(TD) - 1:
        plt.plot(t, epsilon_t * (E1 + E2), '--g')


ax1 = plt.gca()
ax1.set_xlabel('Time')

color1 = 'tab:blue'
ax1.set_ylabel('$\\sigma$', color=color1)
ax1.tick_params(axis='y', labelcolor=color1, colors=color1)
ax1.spines['left'].set_color(color1)
plt.legend()

plt.twinx()
ax2 = plt.gca()
color2 = 'tab:red'
plt.plot(t, epsilon_t, ':r', label='Strain')
ax2.set_ylabel('$\\epsilon$', color=color2)
ax2.tick_params(axis='y', labelcolor=color2, colors=color2)
ax2.spines['right'].set_color(color2)

plt.show()
