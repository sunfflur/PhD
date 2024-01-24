# divisão em blocos

def dataslicing(data, levels=int):
  """
  
    _Slicing Function_
        
    Here we split the original EEG data into several blocks (vectors) of the same shape according to the number of levels defined.
    That means, e.g., 2 levels of slicing will return 5 blocks: the original array + the original array sliced into 4 equaly sized arrays.

  """

  snl = data
  niveis = levels
  N = snl.shape[1]
  totalb = int(((4**(niveis))-1)/3)

  slices = []
  for nivel in range(0,niveis):
    print('nivel:', nivel)
    blocos = 4**nivel
    w = N//blocos
    for b in range(w, N+1, w):
      output = snl[:,b-w:b,:] # 1: [0:375], 2: [375:750], 3: [750:1125], 4:[1125:1500]
      slices.append(output)
  return slices