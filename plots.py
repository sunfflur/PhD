
import numpy as np
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft, fftshift
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn



def hartley_fourier(signal, stimulus_frequency, sampling_frequency, freq_format=True):
  input_signal=signal
  #input_signal = signal
  N = input_signal.shape[0]
  #t = jnp.linspace(0, 6, N) #/sampling_frequency
  t = np.arange(0,N) / sampling_frequency
  #frequencies = jnp.fft.fftfreq(N) * sampling_frequency
  frequencies = np.arange(0,N) * (sampling_frequency/N)
  # Calculate the Fourier transform of the signal
  fft_signal = fft(input_signal)

  # Calculate the magnitude spectrum (absolute values of the Fourier coefficients)
  #magnitude_spectrum0 = filtro_CAR(jnp.abs(fft_signal))
  magnitude_spectrum0 = jnp.abs(fft_signal)
  print(magnitude_spectrum0[:sampling_frequency//2].shape)

  # Calculate the Hartley transform from the real and imaginary parts of the Fourier transform
  hartley_transform0 = jnp.real(fft_signal) - jnp.imag(fft_signal)
  hartley_transform0 = jnp.abs(hartley_transform0)
  #hartley_transform0 = filtro_CAR(hartley_transform0)

  #f,Pxx = welch(input_signal,sampling_frequency)

  if freq_format == True:
    # Create subplots
    fig = sp.make_subplots(rows=1, cols=3, subplot_titles=("Input signal", "Fourier Transform", "Hartley Transform"))

    color0 = dict(color='black')
    color1 = dict(color='hotpink')
    color2 = line=dict(color='limegreen')

    # Add the time-domain plot to the first subplot
    fig.add_trace(go.Scatter(x=t, y=input_signal, mode='lines', name='Input Signal', line=color0), row=1, col=1)
    fig.update_xaxes(title_text="Time [s]", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)

    # Add the frequency-domain plot to the second subplot
    #sampling_freq = N  # Assume a sampling frequency equal to the number of points
    #frequencies = jnp.fft.fftfreq(N) * sampling_freq
    fig.add_trace(go.Scatter(x=frequencies, y=magnitude_spectrum0[:N//2]/N, mode='lines', name='Fourier Transform', line=color0), row=1, col=2) #[:N//2]
    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)

    # Add the Hartley transform plot to the third subplot
    fig.add_trace(go.Scatter(x=frequencies, y=hartley_transform0[:N//2]/N, mode='lines', name='Hartley Transform', line=color0), row=1, col=3) #[:N//2]
    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=3)
    fig.update_yaxes(title_text="Magnitude", row=1, col=3)

    # Add text label at 15 Hz
    plots=[2,3]
    for i in plots:
      fig.add_annotation(
          text=f"{stimulus_frequency} Hz",  # Text to display
          x=stimulus_frequency,          # X-coordinate of the text label
          y=1,           # Y-coordinate of the text label
          showarrow=True,  # Show an arrow pointing to the point
          arrowhead=2,     # Arrowhead style
          arrowsize=1,     # Arrow size
          arrowwidth=1,    # Arrow width
          arrowcolor="red",  # Arrow color
          font=dict(size=12, color="red"),  # Text font size and color
          row=1, col=i
      )

    # Add a red dashed vertical line at x=15 Hz
    vertical_line = go.layout.Shape(
        type="line",
        x0=stimulus_frequency,  # X-coordinate of the line
        x1=stimulus_frequency,  # X-coordinate of the line
        y0=0,   # Starting Y-coordinate
        y1=1,   # Ending Y-coordinate (you can adjust this as needed)
        line=dict(color="red", width=1.5, dash="dash"),  # Red dashed line
    )
    fig.add_shape(vertical_line, row=1, col=2)
    fig.add_shape(vertical_line, row=1, col=3)

    # Update the layout and display the figure
    fig.update_layout(title='', template='plotly_white')
    fig.show()