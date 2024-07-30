import os
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pyvisq import FractionalSLS
from pyvisq.utils import sweep, map_dict

A = 1
B = 0.35  # 0 =< B < A =< 1
A, B = max(A, B), min(A, B)

E1 = 10.0
E2 = E1  # for simplicity
T1 = 10
T2 = 1
CA = E1 * T1 ** A
CB = E2 * T2 ** B
EK = 1.0


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
I = 0.5
D = 0.1
L = 10
config_test = {
    'I': I,
    'D': D,
    'L': L,
}
x_to_y = ('E2', 'EK')
values = [0.5, 1, 2, 10]
cmap = plt.get_cmap('viridis_r')
norm = Normalize(vmin=min(values), vmax=max(values))
sm = ScalarMappable(norm=norm, cmap=cmap)
fig = plt.figure(figsize=(4, 3))

sls_base = FractionalSLS(config_full)
sls_base.set_test(config_test)
sweep_dict = sweep(sls_base, x_to_y, values)

for val, sls in sweep_dict.items():
    sls.run()

    plt.plot(sls.time, sls.stress, color=sm.to_rgba(val),
             label=f"${map_dict[x_to_y[0]]} / {map_dict[x_to_y[1]]}$=" + str(val))

plt.xlabel('Time')
plt.ylabel('$\\sigma$')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('../data/sweep/', str(x_to_y[0]) +
            '_to_' + str(x_to_y[1]) + '.png'), dpi=300)
plt.show()
