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

    The pooling type can be sum, mean or max.
        
"""

def datapooling(data, axis, width, pooling_type: str):
    size = data.shape[2]
    p = []
    for r in range(width, (size//2)+1, width):
      ring1d = create_square(size, r, width).reshape(1,1,data.shape[2],1)
      mult = jnp.multiply(ring1d, data)
      mult = mult.at[mult==0].set(jnp.nan)
      if pooling_type == 'Sum':
        data_pool = jnp.nansum(mult, axis=axis)
      elif pooling_type == 'Mean':
        data_pool = jnp.nanmean(mult, axis=axis)
      elif pooling_type == 'Max':
        data_pool = jnp.nanmax(mult, axis=axis)
      p.append(data_pool)
    return jnp.stack(p, axis=axis)

