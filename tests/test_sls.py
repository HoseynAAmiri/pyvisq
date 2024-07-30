import os

import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pyvisq import SLS
from pyvisq.utils import map_dict

E1 = 1.0
T1 = 1.0
EK = 10.0


config_full = {
    'E1': E1,
    'T1': T1,
    'EK': EK,
}

I = 0.1
D = 0.1
L = 10
config_test = {
    'I': I,
    'D': D,
    'L': L,
}
x_to_y = ('I', 'D')
values = [1, 2, 5, 10, 20]
cmap = plt.get_cmap('viridis_r')
norm = Normalize(vmin=min(values), vmax=max(values))
sm = ScalarMappable(norm=norm, cmap=cmap)
fig_approach = plt.figure(figsize=(4, 3))
fig_dwell = plt.figure(figsize=(4, 3))
sls = SLS(config_full)
for val in values:
    I = val * D
    config_test['I'] = I
    sls.set_test(config_test)
    sls.trigger_force(5.0)

    cruve = sls.get_approach()
    strain, stress = cruve['strain'], cruve['stress']
    plt.figure(fig_approach)
    plt.plot(strain, stress, color=sm.to_rgba(val),
             label=f"${map_dict[x_to_y[0]]} / {map_dict[x_to_y[1]]}$=" + str(val))

    cruve = sls.get_dwell()
    time, stress = cruve['time'], cruve['stress']
    plt.figure(fig_dwell)
    plt.plot(time-time[0], stress, color=sm.to_rgba(val),
             label=f"${map_dict[x_to_y[0]]} / {map_dict[x_to_y[1]]}$=" + str(val))


plt.figure(fig_approach)
plt.xlabel('$\\epsilon$')
plt.ylabel('$\\sigma$')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('../data/trigger/',
            f'approach_SLS_{x_to_y[0]}_to_{x_to_y[1]}_approach.png'), dpi=300)

plt.figure(fig_dwell)
plt.xlabel('Time')
plt.ylabel('$\\sigma$')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('../data/trigger/',
            f'dwell_SLS_{x_to_y[0]}_to_{x_to_y[1]}_dwell.png'), dpi=300)
