import jax.numpy as jnp

"""
    _Frequency Pooling_

    Here we perform the frequency pooling stage, wich means the data is grouped according to the frequency band width chosen.
    The minimun value here is 1.
        
"""
# data now has shape (16, 4, 1000, 6)
def datapooling(data, axis, width, pooling_type):
  steps = jnp.array(range(width, data.shape[2]+1, width))
  p = []
  for i in steps:
    data_steps = data.at[:,:,i-width:i,:].get()
    if pooling_type == 'sum':
      data_pool = jnp.sum(data_steps, axis=axis) # take the sum in axis 2 :> 1000
    elif pooling_type == 'mean':
      data_pool = jnp.mean(data_steps, axis=axis) # take the mean in axis 2
    elif pooling_type == 'max':
      data_pool = jnp.max(data_steps, axis=axis) # take the max in axis 2
    p.append(data_pool)
  return jnp.stack(p, axis=axis)