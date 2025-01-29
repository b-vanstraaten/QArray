###############
GUI
###############

QArray comes with a GUI that allows you to interact with the DotArray class. So rather than writing code, you can use the GUI to create the capacitance matrices and generate the gate voltage arrays. You can then use the GUI to calculate the charge configuration of the quantum dot system at each of these voltage configurations and plot the output.
Below is an example of how to use the GUI to explore the charge stability diagram of a quadruple quantum dot.


.. code:: python

    from qarray import DotArray

    Cdd = [
        [0., 0.3, 0.05, 0.01],
        [0.3, 0., 0.3, 0.05],
        [0.05, 0.3, 0., 0.3],
        [0.01, 0.05, 0.3, 0]
    ]
    Cgd = [
        [1., 0.2, 0.05, 0.01],
        [0.2, 1., 0.2, 0.05],
        [0.05, 0.2, 1., 0.2],
        [0.01, 0.05, 0.2, 1]
    ]

    # setting up the constant capacitance model_threshold_1
    model = DotArray(
        Cdd=Cdd,
        Cgd=Cgd,
        charge_carrier='h', T=0., threshold=1.,
    )
    model.run_gui()

Running this code will produce the terminal output:

.. code:: bash

    Starting the server at http://localhost:9000

Simply click on the link to open the GUI in your browser. The GUI will allow you to interact with the DotArray class and explore the charge stability diagram of a quadruple quantum dot.

From the GUI, you can:

- Edit the capacitance matrices for the quantum dot system.
- Edit the 2D gate voltage scan parameters, such as the gate voltages, the number of points in the scan, and the range of the scan.
- Run the simulation in the open or closed regime (specifying the number of charge carriers in the system).

The GUI will then plot the charge stability diagram of the quantum dot system at each gate voltage configuration.
In addition it labels the charge states with the number of charge carriers in each dot. The GUI output of this code looks like this:

|GUI|

.. |GUI| image:: ./figures/GUI.jpg