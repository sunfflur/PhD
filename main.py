
import tensorflow as tf
import numpy as np
from jax.numpy.fft import fft, ifft, fftshift
import jax.numpy as jnp
from flax import linen as nn

from dataloader import dataloader
from plots import hartley_fourier
datapath = '/content/drive/MyDrive/Doutorado/Dados/Tsinghua BCI Lab - Benchmark Dataset/'

# loading
input_signal15, labels15 = dataloader(subject=1, electrode=False, stimulus_frequency=15, sampling_frequency=250, trial=False, path=datapath)
input_signal10, labels10 = dataloader(subject=1, electrode=False, stimulus_frequency=10, sampling_frequency=250, trial=False, path=datapath)

# plotting
sinal_teste, labels = dataloader(subject=1, electrode=61, stimulus_frequency=15, sampling_frequency=250, trial=1, path=datapath)
hartley_fourier(signal=np.reshape(sinal_teste, (1000)), stimulus_frequency=15, sampling_frequency=250)

input10, l10 = dataloader(subject=1, electrode=61, stimulus_frequency=10, sampling_frequency=250, trial=1, path=datapath)
hartley_fourier(signal=np.reshape(input10, (1000)), stimulus_frequency=10, sampling_frequency=250)