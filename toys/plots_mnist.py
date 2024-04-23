import plotly.graph_objs as go
import plotly.subplots as sp
import jax.numpy as jnp

def plot_mnist(data, label, dhtdata, sampling_frequency):
    color1 = dict(color='rgb(70, 130, 180)')
    color2 = dict(color='rgb(0, 0, 139)')
    color3 = dict(color='rgb(70, 130, 180)')

    # Sampling frequency
    fs = sampling_frequency  # Hz
    # Create time array
    time = jnp.arange(len(data)) / fs
    frequency = jnp.arange(0,len(data)) * (fs/len(data))

    # Assuming first_row_with_label_2 is your filtered row
    fig = sp.make_subplots(rows=3, cols=1)

    for col, value in data.items():
        fig.add_trace(go.Scatter(x=[col], y=[value], mode='markers', name=col), row=1, col=1)
    fig.update_xaxes(title_text="Channels", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        
    fig.add_trace(go.Scatter(x=time, y=data.values, mode='markers', line=color2), row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)

    fig.add_trace(go.Scatter(x=frequency, y=dhtdata.reshape(dhtdata.shape[2]), mode="lines", line=color3), row=3, col=1)
    fig.update_xaxes(title_text="Frequency [Hz]", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=3, col=1)

    fig.update_traces(showlegend=False)
    fig.update_layout(height=700, title=f'Data for Label {int(label.iloc[0])}', template="plotly_white")
    fig.show()