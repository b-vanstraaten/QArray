"""
Author: b-vanstraaten
Date: 04/09/2024
"""
import numpy as np

from qarray import DotArray, charge_state_to_scalar, charge_state_changes
import matplotlib.pyplot as plt

Cdd = [
    [0., 0.1, 0.3, 0.1],
    [0.1, 0., 0.1, 0.3],
    [0.3, 0.1, 0.0, 0.0],
    [0.1, 0.3, 0.0, 0]
]

Cgd = [
    [1., 0., 0.00, 0.0, 0.0],
    [0.0, 1., 0.00, 0.00, 0.0],
    [0.8, 0.2, 1.0, 0.0, 0.0],
    [0.2, 0.8, 0.0, 1.0, 0.0]
]


# setting up the constant capacitance model_threshold_1
model = DotArray(
    Cdd=Cdd,
    Cgd=Cgd,
    charge_carrier='h',
)

# model.run_gui()


vg = model.gate_voltage_composer.do2d('vP1', -3, 3, 400, 'vP2', -3, 3, 400)

vg_top = vg + model.optimal_Vg([0, 0, -3, -3])
vg_left = vg + model.optimal_Vg([0.0, -3, 0.5, -3])
vg_all = vg + model.optimal_Vg([0, 0, 0.5, 0.4])



n_all = model.ground_state_open(vg_all)
z_all = charge_state_changes(n_all)

z_all = np.where(z_all, 1, 0.)
# z_all[z_all == False] = np.nan


plt.imshow(charge_state_to_scalar(n_all), extent=[-3, 3, -3, 3], origin='lower')
plt.imshow(1 - z_all, extent=[-3, 3, -3, 3], origin='lower', alpha=0.1, cmap='gray')
plt.show()

