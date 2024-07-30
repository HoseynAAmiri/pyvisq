import os

import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pyvisq import FractionalSLS

A = 1
B = 0.3  # 0 =< B < A =< 1
A, B = max(A, B), min(A, B)

E1 = 10.0
E2 = 10.0  # for simplicity
T1 = 10.0
T2 = 1.0
CA = E1 * T1 ** A
CB = E2 * T2 ** B
EK = 10.0
mode = 'fast'

map_dict = {
    'A': '\\alpha',
    'B': '\\beta',
    'E1': 'E_1',
    'E2': 'E_2',
    'T1': '\\tau_1',
    'T2': '\\tau_2',
    'EK': 'E_K',
    'CA': 'c_{\\alpha}',
    'CB': 'c_{\\beta}',
    'I': '\\epsilon',
    'D': 't_{ind}',
    'L': 't_{dwell}',
}
config_full = {
    'A': A,
    'B': B,
    'E1': E1,
    'E2': E2,
    'T1': T1,
    'T2': T2,
    'EK': EK,
}
config_direct = {
    'A': A,
    'B': B,
    'CA': CA,
    'CB': CB,
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
sls = FractionalSLS(config_full, mode=mode)
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
            f'approach_FractionalSLS_{x_to_y[0]}_to_{x_to_y[1]}_approach.png'), dpi=300)

plt.figure(fig_dwell)
plt.xlabel('Time')
plt.ylabel('$\\sigma$')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('../data/trigger/',
            f'dwell_FractionalSLS_{x_to_y[0]}_to_{x_to_y[1]}_dwell.png'), dpi=300)
