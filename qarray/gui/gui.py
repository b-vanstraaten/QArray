from time import perf_counter
import io
import zipfile
import base64

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from qarray import charge_state_to_scalar, charge_state_changes
from .helper_functions import create_gate_options, n_charges_options, unique_last_axis, plot_options


def run_gui(model, port=9000, run=True, print_compute_time=True, initial_dac_values=None):
    """
    Create the GUI for the DotArray model.

    Parameters
    ----------
    model : DotArray
    port : int
    run : bool
    print_compute_time : bool
    initial_dac_values : None or array-like
        Optional initial gate values.
    """

    app = dash.Dash(__name__)

    n_dot = model.n_dot
    n_gate = model.n_gate

    # Create the gate options
    gate_options = create_gate_options(model.n_gate, model.n_dot)

    # Convert the matrices to DataFrames for display in the tables
    Cdd = pd.DataFrame(model.Cdd, dtype=float, columns=[f'D{i + 1}' for i in range(n_dot)])
    Cgd = pd.DataFrame(model.Cgd, dtype=float, columns=[f'P{i + 1}' for i in range(n_gate)])

    Cdd[''] = [f'D{i + 1}' for i in range(n_dot)]
    Cgd[''] = [f'D{i + 1}' for i in range(n_dot)]

    # Make the '' column the first column
    Cdd = Cdd[[''] + [col for col in Cdd.columns if col != '']]
    Cgd = Cgd[[''] + [col for col in Cgd.columns if col != '']]

    virtual_gate_matrix = model.compute_optimal_virtual_gate_matrix()
    virtual_gate_matrix = np.round(virtual_gate_matrix, 3)
    virtual_gate_matrix = pd.DataFrame(
        virtual_gate_matrix,
        dtype=float,
        columns=[f'vP{i + 1}' for i in range(n_dot)]
    )

    if initial_dac_values is None:
        initial_dac_values = np.zeros(n_gate)
    else:
        initial_dac_values = np.round(initial_dac_values, 3)

    app.layout = html.Div([
        # First Row: Tables
        html.Div([
            html.Div([
                html.H4("C dot-dot"),
                dash_table.DataTable(
                    id='editable-table1',
                    columns=[{"name": i, "id": i} for i in Cdd.columns],
                    data=Cdd.reset_index().to_dict('records'),
                    editable=True,
                    style_cell_conditional=[
                        {
                            'if': {'column_id': Cdd.columns[0]},
                            'backgroundColor': '#fafafa'  # Light gray color for shading
                        }
                    ]
                )
            ], style={'width': '32%', 'margin-right': '2%'}),

            html.Div([
                html.H4("C gate-dot"),
                dash_table.DataTable(
                    id='editable-table2',
                    columns=[{"name": i, "id": i} for i in Cgd.columns],
                    data=Cgd.reset_index().to_dict('records'),
                    editable=True,
                    style_cell_conditional=[
                        {
                            'if': {'column_id': Cgd.columns[0]},
                            'backgroundColor': '#fafafa'  # Light gray color for shading
                        }
                    ]
                )
            ], style={'width': '32%', 'margin-right': '2%'}),

            html.Div([
                html.H4("Virtual gate matrix"),
                dash_table.DataTable(
                    id='virtual-gate-matrix',
                    columns=[{"name": i, "id": i, "type": "numeric"} for i in virtual_gate_matrix.columns],
                    data=virtual_gate_matrix.reset_index().astype(float).to_dict('records'),
                    editable=True
                )
            ], style={'width': '32%'}),

        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),

        # Second Row: Sweep Options and DAC values
        html.Div([
            html.Div([
                html.H4("X sweep options"),
                dcc.Dropdown(
                    id='dropdown-menu-x',
                    placeholder='X gate',
                    options=gate_options,
                    value='P1'
                ),
                dcc.Input(
                    id='input-scalar-x1',
                    type='number',
                    placeholder='X amplitude',
                    value=5,
                    style={'margin-left': '10px'}
                ),
                dcc.Input(
                    id='input-scalar-x2',
                    type='number',
                    placeholder='X resolution',
                    value=200,
                    style={'margin-left': '10px'}
                ),
            ], style={'width': '24%', 'margin-right': '2%'}),

            html.Div([
                html.H4("Y sweep options"),
                dcc.Dropdown(
                    id='dropdown-menu-y',
                    placeholder='Y gate',
                    options=gate_options,
                    value=f"P{model.n_gate}"
                ),
                dcc.Input(
                    id='input-scalar1',
                    type='number',
                    placeholder='Y amplitude',
                    value=5,
                    style={'margin-left': '10px'}
                ),
                dcc.Input(
                    id='input-scalar2',
                    type='number',
                    placeholder='Y resolution',
                    value=200,
                    style={'margin-left': '10px'}
                ),
            ], style={'width': '24%', 'margin-right': '2%'}),

            html.Div([
                html.H4("DAC values"),
                *[
                    dcc.Input(
                        id=f'dac_{i}',
                        type='number',
                        value=float(initial_dac_values[i]),
                        placeholder=f'P{i}',
                        step=0.1,
                        style={'margin-bottom': '10px', 'display': 'block'}
                    ) for i in range(model.n_gate)
                ]
            ], style={'width': '24%'}),

        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),

        # Third Row: Plot + Options + Save Data
        html.Div([

            html.Div([
                dcc.Graph(
                    id='heatmap',
                    style={'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'}
                )
            ], style={'width': '70%', 'text-align': 'center'}),

            html.Div([
                html.H4("Open/Closed options"),
                dcc.Dropdown(
                    id='dropdown-menu-n-charges',
                    placeholder='Select n charges',
                    options=n_charges_options,
                    value='any'
                ),
                html.H4("Plot options"),
                dcc.Dropdown(
                    id='plot-options',
                    placeholder='Select plot options',
                    options=plot_options,
                    value='changes'
                ),
                html.H4("Automatically update virtual gate matrix"),
                dcc.Dropdown(
                    id='automatically-update-virtual-gate-matrix',
                    placeholder='Auto-update virtual gate matrix',
                    options=[
                        {'label': 'True', 'value': 'True'},
                        {'label': 'False', 'value': 'False'}
                    ],
                    value='True'
                ),

                html.H4("Print charge state"),
                dcc.Dropdown(
                    id='print_charge_state',
                    placeholder='Print charge state True or False',
                    options=[
                        {'label': 'True', 'value': 'True'},
                        {'label': 'False', 'value': 'False'}
                    ],
                    value='True'
                ),

                html.Hr(),

                # --- SAVE DATA BUTTON & DOWNLOAD COMPONENT ---
                html.Button("Save Data", id="save-data", n_clicks=0, style={
                    "fontSize": "20px",  # Increase font size
                    "padding": "15px 30px",  # Increase padding for a larger button
                    "backgroundColor": "grey",
                    "color": "black",
                    "border": "none",
                    "borderRadius": "5px",
                    "cursor": "pointer",
                }),
                dcc.Download(id="download-data"),

            ], style={'width': '28%', 'margin-right': '2%', 'text-align': 'left'}),

        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '20px'})
    ])

    @app.callback(
        (Output('heatmap', 'figure'),
         Output('virtual-gate-matrix', 'data')),
        [Input('editable-table1', 'data'),
         Input('editable-table2', 'data'),
         Input('virtual-gate-matrix', 'data'),
         Input('dropdown-menu-x', 'value'),
         Input('input-scalar-x1', 'value'),
         Input('input-scalar-x2', 'value'),
         Input('dropdown-menu-y', 'value'),
         Input('input-scalar1', 'value'),
         Input('input-scalar2', 'value'),
         Input('dropdown-menu-n-charges', 'value'),
         Input('plot-options', 'value'),
         Input('automatically-update-virtual-gate-matrix', 'value'),
         Input('print_charge_state', 'value'),
         *[Input(f'dac_{i}', 'value') for i in range(model.n_gate)]]
    )
    def update(
            Cdd_data, Cgd_data, virtual_gate_matrix_data,
            x_gate, x_amplitude, x_resolution,
            y_gate, y_amplitude, y_resolution,
            n_charges, plot_choice,
            auto_vg, print_charge_state,
            *dac_values
    ):
        """
        Update the heatmap based on the input values and return the
        updated virtual gate matrix as well.
        """

        if model.T != 0:
            print('Warning the GUI plotting currently only works for T=0. Forcing T=0.')
            model.T = 0

        dac_values = np.array(dac_values)

        if x_gate == y_gate:
            raise ValueError('x_gate and y_gate must be different')

        # Convert table data back to matrices
        try:
            Cdd_df = pd.DataFrame(Cdd_data).drop(columns=['']).set_index('index').astype(float)
            Cgd_df = pd.DataFrame(Cgd_data).drop(columns=['']).set_index('index').astype(float)
        except ValueError:
            print('Error: the capacitance matrices cannot be converted to float.')
            return go.Figure(), virtual_gate_matrix_data

        cdd_matrix = Cdd_df.to_numpy()

        if not np.allclose(cdd_matrix, cdd_matrix.T, equal_nan=True):
            # Replacing NaN with 0, then symmetrize
            cdd_matrix = np.where(np.isnan(cdd_matrix), 0, cdd_matrix)
            print('Warning: Cdd matrix is not symmetric. Taking the average of the upper and lower triangle.')
            cdd_matrix = (cdd_matrix + cdd_matrix.T) / 2

        cgd_matrix = Cgd_df.to_numpy()

        model.update_capacitance_matrices(Cdd=cdd_matrix, Cgd=cgd_matrix)

        # Update the virtual gate matrix automatically if requested
        if auto_vg == 'True':
            updated_vgm = model.compute_optimal_virtual_gate_matrix()
            updated_vgm = np.round(updated_vgm, 3)
            virtual_gate_matrix = pd.DataFrame(
                updated_vgm,
                columns=[f'vP{i + 1}' for i in range(n_dot)]
            )
        else:
            # Use the user-edited matrix as is
            virtual_gate_matrix = pd.DataFrame(virtual_gate_matrix_data)
            # Just in case the user changed dimensions
            virtual_gate_matrix = virtual_gate_matrix.iloc[:, :n_dot]

        model.gate_voltage_composer.virtual_gate_matrix = virtual_gate_matrix.to_numpy()

        # Sweep
        vg = model.gate_voltage_composer.do2d(
            x_gate, -x_amplitude / 2, x_amplitude / 2, x_resolution,
            y_gate, -y_amplitude / 2, y_amplitude / 2, y_resolution
        ) + dac_values[np.newaxis, np.newaxis, :]

        # Compute charge states
        t0 = perf_counter()
        if n_charges == 'any':
            n = model.ground_state_open(vg)
        else:
            n = model.ground_state_closed(vg, n_charges=n_charges)
        t1 = perf_counter()
        if print_compute_time:
            print(f'Time to compute charge states: {t1 - t0:.3f} s')

        # Build 'z' for the heatmap
        if plot_choice in px.colors.named_colorscales():
            z = charge_state_to_scalar(n).astype(float)
            z = np.log2(z + 1)  # optional for better contrast
            cmap = plot_choice
        elif plot_choice == 'changes':
            z = charge_state_changes(n).astype(float)
            cmap = 'greys'
        else:
            raise ValueError(
                f'Unrecognized plot choice "{plot_choice}". '
                f'Either use "changes" or one of {px.colors.named_colorscales()}.'
            )

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            colorscale=cmap,
            showscale=False,  # remove the colorbar
        ))

        x_text = np.linspace(-x_amplitude / 2, x_amplitude / 2, 11).round(3)
        x_tickvals = np.linspace(0, x_resolution, 11)

        y_text = np.linspace(-y_amplitude / 2, y_amplitude / 2, 11).round(3)
        y_tickvals = np.linspace(0, y_resolution, 11)

        fig.update_xaxes(title_text=x_gate, ticktext=x_text, tickvals=x_tickvals)
        fig.update_yaxes(title_text=y_gate, ticktext=y_text, tickvals=y_tickvals)

        charge_states = unique_last_axis(n).astype(int)

        # Optionally annotate the figure with the distinct charge states
        if print_charge_state == 'True':
            if charge_states.shape[0] <= 100:
                for charge_state in charge_states:
                    ix, iy = np.where(np.all(n == charge_state, axis=-1))
                    avg_x = iy.mean()
                    avg_y = ix.mean()
                    label = ','.join(map(str, charge_state))
                    fig.add_annotation(
                        x=avg_x,
                        y=avg_y,
                        text=label,
                        showarrow=False,
                        font=dict(color='black', size=11)
                    )
            else:
                print(f'Skipping charge-state annotation (>100 distinct states).')

        fig.update_layout(
            title='Charge stability diagram',
            xaxis_nticks=4,
            yaxis_nticks=4,
            autosize=False,
            width=600,
            height=600,
        )

        # Return figure + updated Virtual Gate Matrix
        return fig, virtual_gate_matrix.round(3).to_dict('records')

    # ------------------------------------------------------------------
    #  Callback to save data on button click
    # ------------------------------------------------------------------
    @app.callback(
        Output("download-data", "data"),
        Input("save-data", "n_clicks"),
        State("heatmap", "figure"),
        State('dropdown-menu-x', 'value'),
        State('input-scalar-x1', 'value'),
        State('input-scalar-x2', 'value'),
        State('dropdown-menu-y', 'value'),
        State('input-scalar1', 'value'),
        State('input-scalar2', 'value'),
        [State(f'dac_{i}', 'value') for i in range(model.n_gate)],
        prevent_initial_call=True
    )
    def save_data(n_clicks, heatmap_fig,
                  x_gate, x_amplitude, x_resolution,
                  y_gate, y_amplitude, y_resolution,
                  *dac_values):
        """
        Save the z data from the heatmap, as well as the Cdd and Cgd matrices,
        into a single ZIP file containing:
          - z_data.npy
          - Cdd.npy
          - Cgd.npy
          - vg.npy
          - virtual_gate_matrix.npy
        """

        print(f"Save Data Button Clicked: {n_clicks}")  # Debugging

        if n_clicks is None or n_clicks < 1:
            raise PreventUpdate

        dac_values = np.array(dac_values)

        # Sweep
        vg = model.gate_voltage_composer.do2d(
            x_gate, -x_amplitude / 2, x_amplitude / 2, x_resolution,
            y_gate, -y_amplitude / 2, y_amplitude / 2, y_resolution
        ) + dac_values[np.newaxis, np.newaxis, :]

        # Extract z data from the heatmap figure
        z = np.array(heatmap_fig["data"][0]["z"])

        # Prepare ZIP file in-memory
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            # z_data.npy
            z_data_bytes = io.BytesIO()
            np.save(z_data_bytes, z)
            zf.writestr("z_data.npy", z_data_bytes.getvalue())

            # Cdd.npy
            cdd_bytes = io.BytesIO()
            np.save(cdd_bytes, model.cdd)
            zf.writestr("Cdd.npy", cdd_bytes.getvalue())

            # Cgd.npy
            cgd_bytes = io.BytesIO()
            np.save(cgd_bytes, model.cgd)
            zf.writestr("Cgd.npy", cgd_bytes.getvalue())

            # vg.npy
            vg_bytes = io.BytesIO()
            np.save(vg_bytes, vg)
            zf.writestr("vg.npy", vg_bytes.getvalue())

            # virtual_gate_matrix.npy
            virtual_gate_matrix_bytes = io.BytesIO()
            np.save(virtual_gate_matrix_bytes, model.gate_voltage_composer.virtual_gate_matrix)
            zf.writestr("virtual_gate_matrix.npy", virtual_gate_matrix_bytes.getvalue())

        # Encode and return the ZIP file
        zip_bytes = buffer.getvalue()
        b64_zip = base64.b64encode(zip_bytes).decode("utf-8")
        return dict(
            content=b64_zip,
            filename="dotarray_data.zip",
            base64=True,
        )

    # Run the server
    if run:
        print(f'Starting the server at http://localhost:{port}')
        app.run(debug=False, port=port)

    return app