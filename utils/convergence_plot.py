import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np

def convergence_plot(iterations: list, perfs: list):

    n_calls = len(perfs)
    iterations = list(range(1, n_calls + 1))
    mins = [np.min(perfs[:i]) for i in iterations]
    max_mins = max(mins)
    cliped_losses = np.clip(perfs, None, max_mins)
    
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=iterations, y=mins,
                        mode='lines', showlegend=False,
                        ),
                )
    fig.add_trace(go.Scatter(x=iterations, y=cliped_losses,
                        mode='markers', showlegend=False,
                    ),
                )
    fig.update_layout(
        title={
            'text': "Convergence plot",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Number of iterations",
        yaxis_title="Min objective value after n iterations",
    )

    plotly_plot_obj = plot({'data': fig}, output_type='div')

    return plotly_plot_obj