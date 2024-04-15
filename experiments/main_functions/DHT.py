import jax.numpy as jnp
from jax.numpy.fft import fft


def dataDHT(data):
    """
    
        _Discrete Hartley Transform_
        This function implements the DHT based on the Real and Imaginary parts of the DFT.
        
    """    
    # data now has shape (16, 4, 1000, 6)    
    fourier_data = fft(data, axis=2)
    hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_hartley = jnp.abs(hartley_data)
    
    return norm_hartley#.at[:,:,:data.shape[2]//2].get() / (data.shape[2]) # /data.shape[1] #data.shape[1]//2