import jax.numpy as jnp
from jax.numpy.fft import fft


def dataDFT(data):
    """
    
        _Discrete Fourier Transform_
        This function implements the DFT.
        
    """    
        
    fourier_data = fft(data, axis=1)
    #hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_fourier = jnp.abs(fourier_data)
    
    return norm_fourier.at[:,:data.shape[1]//2].get() / (data.shape[1]) 