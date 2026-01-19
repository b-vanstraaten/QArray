from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from qarray import ChargeSensedDotArray, charge_state_changes, TelegraphNoise, \
    WhiteNoise, LatchingModel
from qarray.gui import unique_last_axis


def compute_triple_point(model, vg, charge_states, tol = 1e-2):
    # Calculate matrices c and a
    c = np.stack([
        (Na.T @ cdd_inv @ Na - Nb.T @ cdd_inv @ Nb).squeeze()
        for Na, Nb in combinations(charge_states, r=2)
    ])
    a = np.stack([
        2 * (model.cgd.T @ cdd_inv @ (Na - Nb)).squeeze()
        for Na, Nb in combinations(charge_states, r=2)
    ])

    # Solve for x and calculate error
    pinv = np.linalg.pinv(a)
    x = pinv @ c
    error = np.eye(3) - pinv @ a

    virtual_gate_matrix = -np.linalg.pinv(model.cdd_inv_full @ model.cgd_full)
    # Calculate the direction vector l
    l = error @ virtual_gate_matrix[:, -1]
    l = l / np.linalg.norm(l)

    # Compute the normal of the plane defined by vg points
    p0, p1, p2 = vg[0, 0, :], vg[-1, 0, :], vg[0, -1, :]
    plane_normal = np.cross(p1 - p0, p2 - p0)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    d = np.dot(plane_normal, p0)

    v_interdot_true = point_of_intersection(plane_normal, d, x, l)

    # Grid basis
    origin = vg[0, 0]
    dx = vg[0, 1] - vg[0, 0]
    dy = vg[1, 0] - vg[0, 0]

    # Solve affine system (least squares allows extrapolation)
    A = np.stack([dy, dx], axis=1)   # shape (D, 2)
    b = v_interdot_true - origin     # shape (D,)

    iy, ix = np.linalg.lstsq(A, b, rcond=None)[0]

    # Normalize (NOTE: use -1 for proper normalization)
    ix /= (vg.shape[1] - 1)
    iy /= (vg.shape[0] - 1)

    return [ix, iy]


def point_of_intersection(plane_normal, d, x, n):
    """
    Function to find the point of intersection between a plane and a line.

    Parameters:
    plane_normal (np.ndarray): Normal vector of the plane.
    d (float): Distance of the plane from the origin.
    x (np.ndarray): Point on the line.
    n (np.ndarray): Direction of the line.

    Returns:
    np.ndarray: Point of intersection.
    """

    # Calculate the distance of the point from the plane
    t = (d - np.dot(plane_normal, x)) / np.dot(plane_normal, n)

    # Calculate the point of intersection
    return x + t * n


# defining the capacitance matrices
Cdd = [[0., 0.3], [0.3, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.1, 0.02], [0.2, 1., 0.05]]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.1, 0.01]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# defining a white noise model with an amplitude of 1e-2
white_noise = WhiteNoise(amplitude=1e-2)

# defining a telegraph noise model with p01 = 5e-4, p10 = 5e-3 and an amplitude of 1e-2
random_telegraph_noise = TelegraphNoise(p01=5e-5, p10=5e-3, amplitude=1e-2)

# combining the white and telegraph noise models
combined_noise = white_noise + random_telegraph_noise

latching_model = LatchingModel(
    n_dots=2,
    p_leads=[0.9, 0.9],
    p_inter=[
        [0., 0.9],
        [0.9, 0.],
    ]
)

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.05, T=0, noise_model=combined_noise, latching_model=latching_model
)


# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -2, 2
vy_min, vy_max = -2, 2

model.gate_voltage_composer.virtual_gate_matrix = np.array([
    [1, -0.1, 0],
    [-0.1, 1, 0],
    [-0.1, -0.05, 0]
])
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = model.gate_voltage_composer.do2d('vP1', vy_min, vx_max, 400, 'vP2', vy_min, vy_max, 400)

# centering the voltage sweep on the [0, 1] - [1, 0] interdot charge transition on the side of a charge sensor coulomb peak
vg += model.optimal_Vg([0.5, 0.5, 0.5])

# calculating the output of the charge sensor and the charge state for each gate voltage
z, n = model.charge_sensor_open(vg)

unique_charge_states = unique_last_axis(np.round(n))

upper_triple_points = []
upper_triple_point_locations = []

lower_triple_points = []
lower_triple_point_locations = []

# Precompute common matrices and variables outside the loop
cdd_inv = model.cdd_inv_full[:2, :2]
virtual_gate_matrix = model.gate_voltage_composer.virtual_gate_matrix

for charge_state in unique_charge_states:
    # Determine adjacent charge states
    charge_state_left = charge_state + np.array([1, 0])
    charge_state_down = charge_state + np.array([0, 1])

    # Store the triple point
    triple_point = (charge_state, charge_state_left, charge_state_down)
    voltage = compute_triple_point(model, vg, triple_point)
    upper_triple_points.append(triple_point)
    upper_triple_point_locations.append(voltage)

    charge_state_right = charge_state + np.array([-1, 0])
    charge_state_up = charge_state + np.array([0, -1])

    if np.all(charge_state_right >= 0) and np.all(charge_state_up >= 0):
        # Store the triple point
        triple_point = (charge_state, charge_state_right, charge_state_up)
        voltage = compute_triple_point(model, vg, triple_point)
        lower_triple_points.append(triple_point)
        lower_triple_point_locations.append(voltage)

vg_x = vg[:, :, 0]
vg_y = vg[:, :, 1]
extent_x = (0, 1)
extent_y = (0, 1)

extent = (*extent_x, *extent_y)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(10, 5)

# plotting the charge stability diagram
axes[0].imshow(z, extent=extent, origin='lower', aspect='auto', cmap='hot')
axes[0].set_xlabel('$Vx$')
axes[0].set_ylabel('$Vy$')
axes[0].set_title('$z$')

# plotting the charge sensor output
axes[1].imshow(charge_state_changes(n), extent=extent, origin='lower', aspect='auto', cmap='Greys')
axes[1].set_xlabel('$Vx$')
axes[1].set_ylabel('$Vy$')
axes[1].set_title('$n$')

upper_triple_point_locations = np.array(upper_triple_point_locations)
lower_triple_point_locations = np.array(lower_triple_point_locations)

axes[0].scatter(upper_triple_point_locations[:, 0], upper_triple_point_locations[:, 1], color='blue')
axes[1].scatter(upper_triple_point_locations[:, 0], upper_triple_point_locations[:, 1], color='blue')

axes[0].scatter(lower_triple_point_locations[:, 0], lower_triple_point_locations[:, 1], color='green')
axes[1].scatter(lower_triple_point_locations[:, 0], lower_triple_point_locations[:, 1], color='green')

plt.xlim(*extent_x)
plt.ylim(*extent_y)
plt.show()

import h5py

with h5py.File('triple_points.h5', 'w') as f:

    dset = f.create_dataset('vg', shape = (100, *z.shape))
    dset[0] = z
