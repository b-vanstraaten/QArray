import matplotlib.pyplot as plt
import numpy as np

from qarray import ChargeSensedDotArray
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


print(model.gate_voltage_composer.virtual_gate_matrix)



# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -2, 2
vy_min, vy_max = -2, 2
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = model.gate_voltage_composer.do2d('vP1', vy_min, vx_max, 100, 'vP2', vy_min, vy_max, 100)

# centering the voltage sweep on the [0, 1] - [1, 0] interdot charge transition on the side of a charge sensor coulomb peak
vg += model.optimal_Vg([0.5, 0.5, 0.5])


# calculating the output of the charge sensor and the charge state for each gate voltage
z, n = model.charge_sensor_open(vg)
dz_dV1 = np.gradient(z, axis=0) + np.gradient(z, axis=1)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(10, 5)

# plotting the charge stability diagram
axes[0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='viridis')
axes[0].set_xlabel('$Vx$')
axes[0].set_ylabel('$Vy$')
axes[0].set_title('$z$')

# plotting the charge sensor output
axes[1].imshow(dz_dV1, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='viridis')
axes[1].set_xlabel('$Vx$')
axes[1].set_ylabel('$Vy$')
axes[1].set_title('$\\frac{dz}{dVx} + \\frac{dz}{dVy}$')

plt.savefig('../docs/source/figures/charge_sensing.jpg', dpi=300)
plt.show()

