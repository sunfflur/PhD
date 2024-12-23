import jax.numpy as jnp
from jax.numpy.fft import fft, fftshift


def dataDFT(data):
    """
    
        _Discrete Fourier Transform_
        This function implements the DFT.
        
    """    
    # data now has shape (16, 4, 1000, 6)
    fourier_data = fftshift(fft(data, axis=2), axes=2)
    #hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_fourier = jnp.abs(fourier_data)
    
    return norm_fourier #norm_fourier.at[:,:,:data.shape[2]//2,:].get() / (data.shape[2])

def dataDFT_half(data):
    """
    
        _Discrete Fourier Transform_
        This function implements the DFT. We get only half the DFT based on its symmetry.
        
    """    
    # data now has shape (16, 4, 1000, 6)
    fourier_data = fft(data, axis=2)
    #hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_fourier = jnp.abs(fourier_data)
    
    return norm_fourier.at[:,:,:data.shape[2]//2,:].get() #/ (data.shape[2])