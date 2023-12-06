import numpy as np
import os
from flax import linen as nn
from dataloader import dataloader
from plots import hartley_fourier
datapath = '/content/drive/MyDrive/Doutorado/Dados/Tsinghua BCI Lab - Benchmark Dataset/'
#datapath = 'g:\Meu Drive\Doutorado\Dados\Tsinghua BCI Lab - Benchmark Dataset'
#datapath = '/home/natalia/'
datapath = '/run/user/1002/gvfs/google-drive:host=dac.unicamp.br,user=n232881/0AMgULXMB0MAlUk9PVA/1d4-2xyAIuuocPDNEC50Tzt8tSkNnZsNC/1BEjbn5jRzSNXfQaIyMbbqK-Gg-CEuDh4/'
print(os.path.isdir('/run/user/1002/gvfs/google-drive:host=dac.unicamp.br,user=n232881/0AMgULXMB0MAlUk9PVA/1d4-2xyAIuuocPDNEC50Tzt8tSkNnZsNC/1BEjbn5jRzSNXfQaIyMbbqK-Gg-CEuDh4/'))
# loading

"""
    Load data from from different stimulus frequency: 8, 10, 12 and 15 Hertz and
    choosing 16 electrodes (O1, O2, Oz, POz, Pz, PO3, PO4, PO7, PO8, P1, P2, Cz, C1, C2, CPz, FCz),
    following Vargas et. al (2022).

"""
stimulif = [8,10,12,15]
input_signal15, labels15 = dataloader(subject=1, electrode=False, stimulus_frequency=stimulif, trial=False, path=datapath)
input_signal10, labels10 = dataloader(subject=1, electrode=False, stimulus_frequency=10, trial=False, path=datapath)

# plotting
sinal_teste, labels = dataloader(subject=1, electrode=61, stimulus_frequency=15, trial=1, path=datapath)
hartley_fourier(signal=np.reshape(sinal_teste, (1000)), stimulus_frequency=15, sampling_frequency=250)

input10, l10 = dataloader(subject=1, electrode=61, stimulus_frequency=10, trial=1, path=datapath)
hartley_fourier(signal=np.reshape(input10, (1000)), stimulus_frequency=10, sampling_frequency=250)