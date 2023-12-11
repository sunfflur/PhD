import jax.numpy as jnp

"""
    _Frequency Pooling_

    Here we perform the frequency pooling stage, wich means the data is grouped according to the frequency band width chosen.
    The minimun value here is 1.
        
"""

def datapooling(data, axis, width):
  steps = jnp.array(range(width, data.shape[1]+1, width))
  p = []
  for i in steps:
    data_steps = data.at[:,i-width:i,:].get()
    data_sum = jnp.sum(data_steps, axis=axis) # soma no eixo 1 :> 1500
    p.append(data_sum)
    # saída esperada: (64, 1500/5, 6)
  return jnp.stack(p, axis=axis)