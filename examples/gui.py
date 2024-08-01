from qarray import DotArray

Cdd = [
    [0., 0.2, 0.05, 0.01],
    [0.2, 0., 0.2, 0.05],
    [0.05, 0.2, 0.0, 0.2],
    [0.01, 0.05, 0.2, 0]
]

Cgd = [
    [1., 0.1, 0.05, 0.01],
    [0.1, 1., 0.1, 0.05],
    [0.05, 0.1, 1., 0.1],
    [0.01, 0.05, 0.1, 1]
]


# setting up the constant capacitance model_threshold_1
model = DotArray(
    Cdd=Cdd,
    Cgd=Cgd,
    charge_carrier='h',
)

model.run_gui()
