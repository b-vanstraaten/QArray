import matplotlib.pyplot as plt
import numpy as np

from qarray import ChargeSensedDotArray, charge_state_changes, dot_occupation_changes
from qarray.noise_models import WhiteNoise, TelegraphNoise, NoNoise
from time import perf_counter

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.1, 0.05], [0.1, 1., 0.05]]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.05, 0.00]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.05, T=0,
    implementation='rust', algorithm='default'
)


virtual_gate_matrix = model.gate_voltage_composer.virtual_gate_matrix

pert = np.random.uniform(0, 0.0, size=virtual_gate_matrix.shape) + 1.

model.gate_voltage_composer.virtual_gate_matrix = virtual_gate_matrix * pert


# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -1, 1
vy_min, vy_max = -1, 1
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = model.gate_voltage_composer.do2d('vP1', vy_min, vx_max, 100, 'vP2', vy_min, vy_max, 100)

# centering the voltage sweep on the [0, 1] - [1, 0] interdot charge transition on the side of a charge sensor coulomb peak
vg += model.optimal_Vg([0.5, 0.5, 0.5])



# calculating the output of the charge sensor and the charge state for each gate voltage
z, n = model.charge_sensor_open(vg)
z = z.squeeze()

change_in_dot_0 = charge_state_changes(n, 0)
change_in_dot_1 = charge_state_changes(n, 1)


i = np.logical_and(change_in_dot_0, change_in_dot_1)

v = np.logical_xor(change_in_dot_0, i)
h = np.logical_xor(change_in_dot_1, i)

no_transition = np.logical_not(charge_state_changes(n))

fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
axes = axes.flatten()
fig.set_size_inches(10, 2)

# plotting the charge stability diagram
axes[0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='equal', cmap='viridis')
axes[0].set_xlabel('$Vx$')
axes[0].set_ylabel('$Vy$')
axes[0].set_title('$z$')


variables = {
    'horizontal': h,
    'vertical': v,
    'interdot': i,
    'no_transition': no_transition
}

offset = 1
for index, (label, value) in enumerate(variables.items()):
    axes[offset + index].imshow(value, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='equal', cmap='Greys')
    axes[offset + index].set_xlabel('$Vx$')
    axes[offset + index].set_ylabel('$Vy$')
    axes[offset + index].set_title(label)

    axes[offset + index].set_xlabel('$Vx$')
    axes[offset + index].set_ylabel('$Vy$')

fig.tight_layout()

plt.savefig('../docs/source/figures/charge_sensing.jpg', dpi=300)
plt.show()

labels = np.stack(
    [h, v, i, no_transition], axis = 0
).astype(int)

