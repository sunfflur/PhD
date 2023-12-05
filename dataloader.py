import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat,savemat
seed=11



def dataloader(subject, electrode, stimulus_frequency, sampling_frequency, trial, path):
  #escolha do sujeito cujos dados vamos processar/analisar
  suj = subject
  filename = "S" + str(suj) + ".mat"

  #escolha da sessão
  #trial = trial

  #escolha da frequência de estímulo (pelo valor em Hz)
  #freq_est = stimulus_frequency

  #carrega o vetor de frequências de estímulo na ordem correspondente aos índices
  data_freq = loadmat(path+'Freq_Phase.mat')
  #isola o vetor de frequências
  freqs = data_freq['freqs']
  #print(freqs)
  #encontra o índice da frequência selecionada
  ii = np.where(freqs==stimulus_frequency)

  #carrega os dados de EEG
  data_eeg = loadmat(path+filename)
  data = data_eeg['data']

  start = 250 #250
  end = -250 #-250
  #isola os registros de EEG dos L eletrodos daquele sujeito, para aquela freq. de estímulo e para aquela sessão
  if type(trial) == int:
    #EEG = filtro_CAR(np.reshape(data[:,start:end,ii[1],trial],(64,1500-(start-end),1)))
    EEG = np.reshape(data[:,start:end,ii[1],trial],(64,1500-(start-end),1))
  else:
    #EEG = filtro_CAR(np.reshape(data[:,start:end,ii[1],:],(64,1500-(start-end),6)))
    EEG = np.reshape(data[:,start:end,ii[1],:],(64,1500-(start-end),6))

  #isola os registros de EEG dos L eletrodos daquele sujeito para aquela sessão
  #EEG_ss = np.reshape(data[:,:,:,trial],(64,1500))
  #dic = {"data": EEG_ss}
  #savemat('EEG.mat',dic)

  #frequência de amostragem
  # = sampling_frequency
  #duração do sinal (em amostras)
  #N = EEG.shape[1] # 1500 pontos

  # filtro eletrodos do cortex visual
  #visual_cortex = {52:'PO7', 54:'PO3', 55:'POz', 56:'PO4', 58:'PO8',60:'O1', 61:'Oz', 62:'O2'}
  #visual_cortex = {60:'O1', 61:'Oz', 62:'O2'}

  if type(electrode) == int:
    #índice correspondente ao eletrodo que queremos analisar (61 = eletrodo Oz, 55 = POz)
    id_ele = electrode #61
    #captura do sinal de EEG
    sinal = EEG[id_ele,:]
    labels = jnp.ones((1, sinal.shape[1]))*stimulus_frequency
  else:
    #sinal = EEG[:,:]
    sinal = EEG[:, :] #list(visual_cortex)
    labels = jnp.ones((sinal.shape[0], sinal.shape[2]))*stimulus_frequency
  return sinal, labels