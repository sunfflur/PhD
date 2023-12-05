

# divisão em blocos

def slicing(data, levels=int):
  snl = data
  niveis = levels
  N = snl.shape[1]
  totalb = int(((4**(niveis))-1)/3)

  new = []
  for nivel in range(0,niveis):
    blocos = 4**nivel
    print(blocos)
    w = N//blocos
    for b in range(w, N+1, w):
      #print(b,w)
      output = snl[:,b-w:b,:] # 1: [0:375], 2: [375:750], 3: [750:1125], 4:[1125:1500]
      print(output.shape)
      new.append(output)
    print('len', len(new))
  return new