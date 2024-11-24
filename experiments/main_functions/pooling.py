import jax.numpy as jnp


def create_square(size=None, radius=None, width=None):
  image = jnp.zeros(size)
  center = (size-1) / 2
  linhas=jnp.arange(0,size,1).reshape(size,1) #u
  condition_1=jnp.abs(linhas-center) # |x1-x2|
  maximo=condition_1 # max(|x1-x2|,|y1-y2|) # chebyshev
  desigualdade=jnp.less_equal(maximo,radius)
  desigualdade_2=jnp.greater(maximo,radius-width) # max <= r
  return jnp.logical_and(desigualdade,desigualdade_2).astype(jnp.float32)


"""
    _Frequency Pooling_

    Here we perform the frequency pooling stage, wich means the data is grouped according to the frequency band width chosen.
    The minimun value here is 1.
        
"""

#data = np.asarray([[10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10]]).T
#print("data", data.shape)

def datapooling(data, axis, width, pooling_type):
    size = data.shape[2]
    p = []
    for r in range(width, (size//2)+1, width):
      ring1d = create_square(size, r, width).reshape(1,1,data.shape[2],1)
      mult = jnp.multiply(ring1d, data)
      if pooling_type == 'sum':
        data_pool = jnp.sum(mult, axis=axis) #keepdims=True
      p.append(data_pool)
    return jnp.stack(p, axis=axis)















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
"""



