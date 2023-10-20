"""
This script recreates figure 3 from the paper https://www.nature.com/articles/s41565-020-00816-w#Sec19
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes)

nearest_coupling = 0.1
diagonal_coupling = 0.05
far_coupling = 0.01

cdd_non_maxwell = [
    [0., diagonal_coupling, nearest_coupling, diagonal_coupling, far_coupling],
    [diagonal_coupling, 0., nearest_coupling, far_coupling, diagonal_coupling],
    [nearest_coupling, nearest_coupling, 0., nearest_coupling, nearest_coupling],
    [diagonal_coupling, far_coupling, nearest_coupling, 0, diagonal_coupling],
    [far_coupling, diagonal_coupling, nearest_coupling, diagonal_coupling, 0.]
]

cross_talk = 0.10
cgd_non_maxwell = np.array([
    [1., 0, cross_talk, 0., 0.],
    [0, 2.3, cross_talk, 0., 0.],
    [0, 0, 1., 0, 0],
    [0., 0., cross_talk, 2.3, 0.],
    [0., 0., cross_talk, 0., 1.]
]).T


model = DotArray(
    cdd_non_maxwell=cdd_non_maxwell,
    cgd_non_maxwell=cgd_non_maxwell,
    core='rust',
    threshold=1.,
)

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
virtual_gate_matrix = np.linalg.pinv(-model.cdd_inv @ model.cgd)
virtual_gate_origin = np.zeros(shape=(model.n_gate,))

voltage_composer = GateVoltageComposer(
    n_gate=model.n_gate,
    virtual_gate_matrix=virtual_gate_matrix,
    virtual_gate_origin=virtual_gate_origin
)

N = 300
# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -1.9, 1.9
vy_min, vy_max = -2.3, 2.8

vg = voltage_composer.do2d_virtual(1, vy_min, vx_max, N, 3, vy_min, vy_max, N)

# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg_lt = voltage_composer.do2d_virtual(1, vx_min, vx_max, N, 0, vy_min, vy_max, N)
vg_lb = voltage_composer.do2d_virtual(1, vx_min, vx_max, N, 4, -vy_min, -vy_max, N)
vg_rt = voltage_composer.do2d_virtual(3, -vx_min, -vx_max, N, 0, vy_min, vy_max, N)
vg_rb = voltage_composer.do2d_virtual(3, -vx_min, -vx_max, N, 4, -vy_min, -vy_max, N)

vg = vg_lt + vg_lb + vg_rt + vg_rb
scale = 1.
shift = -0.2

vg = vg + voltage_composer.do1d(2, shift - scale, shift + scale, N)[:, np.newaxis, :]

# creating the figure and axes
# fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
# fig.set_size_inches(15, 4)

t0 = time.time()
n = model.ground_state_closed(vg, 5)
t1 = time.time()
print(f'Ground state calculation took {t1 - t0:.2f} seconds')

z = dot_occupation_changes(n)

names = ['T', 'L', 'M', 'R', 'B']

# for i in range(5):
#     axes[i].set_title(names[i])
#     axes[i].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='Greys', alpha=1.)
#     axes[i].imshow(n[..., i], extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto',
#                    interpolation='None', vmin=0, vmax=5, alpha=0.9
#                    , cmap=parula_map)
#
# plt.figure()

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(8, 4)

# my_data = np.genfromtxt('./closed_array.csv', delimiter=',')

# ax[0].imshow(my_data, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='Greys', alpha=1.,
#              interpolation='spline16')
ax[1].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='Greys', alpha=1.,
             interpolation='spline16')
plt.savefig('5_dots.pdf', bbox_inches='tight')
plt.show()
