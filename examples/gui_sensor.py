"""
Author: b-vanstraaten
Date: 07/12/2024
"""
from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, charge_state_changes, LatchingModel


# # # defining the capacitance matrices for a double dot
# Cdd = [[0., 0.1],
#        [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
# Cgd = [[1., 0.1, 0.05],
#        [0.1, 1., 0.05]]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
# Cds = [[0.05, 0.03]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
# Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots
# initial_dac_values = [0, 0.0, 0.5]


# defining the capacitance matrices for a linear array of 4 dots, uncomment this block and comment the previous block to switch
# with the sensor on the left hand side
Cdd = [
    [0., 0.2, 0.05, 0.01],
    [0.2, 0., 0.2, 0.05],
    [0.05, 0.2, 0.0, 0.2],
    [0.01, 0.05, 0.2, 0]
]
Cgd = [
    [1., 0.1, 0.05, 0.01, 0.1],
    [0.1, 1., 0.1, 0.05, 0.002],
    [0.05, 0.1, 1., 0.1, 0.001],
    [0.01, 0.05, 0.1, 1, 0.000]
]

Cds = [[0.1, 0.05, 0.001, 0.000]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.1, 0.04, 0.003, 0.000, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots
initial_dac_values = [0, 0, 0., 0.0, 0.5]


noise_model = WhiteNoise(amplitude=1e-2)

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.1,
    T=0, # mK
    noise_model=noise_model,
)
initial_virtual_gate_matrix = model.compute_optimal_sensor_virtual_gate_matrix()
model.gui(initial_dac_values = initial_dac_values, initial_virtual_gate_matrix = initial_virtual_gate_matrix)