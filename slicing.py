# divisão em blocos

def dataslicing(data, levels=int):
  snl = data
  niveis = levels
  N = snl.shape[1]
  totalb = int(((4**(niveis))-1)/3)

  slices = []
  for nivel in range(0,niveis):
    blocos = 4**nivel
    print(blocos)
    w = N//blocos
    for b in range(w, N+1, w):
      output = snl[:,b-w:b,:] # 1: [0:375], 2: [375:750], 3: [750:1125], 4:[1125:1500]
      print(output.shape)
      slices.append(output)
    print('len', len(slices))
  return slices