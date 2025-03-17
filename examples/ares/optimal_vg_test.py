from qarray import DotArray


def system_setup():
    Cdd = [
        [0., 0.3, 0.05],
        [0.3, 0., 0.3],
        [0.05, 0.3, 0]
    ]  # an (n_dot, n_dot) array of the capacitive coupling between dots

    Cgd = [
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots

    # creating the model
    model = DotArray(
        Cdd=Cdd, Cgd=Cgd
    )
    return model


model = system_setup()
v1 = model.optimal_Vg([1, 1, 1])
v2 = model.optimal_Vg([1, 100, 1])
