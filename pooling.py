import jax.numpy as jnp

# Agrupamento

def datapooling(data, axis, width):
  steps = jnp.array(range(width, data.shape[1]+1, width))
  p = []
  for i in steps:
    data_steps = data.at[:,i-width:i,:].get()
    data_sum = jnp.sum(data_steps, axis=axis) # soma no eixo 1 :> 1500
    p.append(data_sum)
    # saída esperada: (64, 1500/5, 6)
  return jnp.stack(p, axis=axis)