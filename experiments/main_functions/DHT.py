import jax
import jax.numpy as jnp
from jax.numpy.fft import fft, fftshift

@jax.jit
def dataDHT(data):
    """
    
        _Discrete Hartley Transform_
        This function implements the DHT based on the Real and Imaginary parts of the DFT.
        
    """    
    # data now has shape (16, 4, 1000, 6)    
    fourier_data = fftshift(fft(data, axis=2), axes=2)
    hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_hartley = jnp.abs(hartley_data)
    
    return norm_hartley#.at[:,:,:data.shape[2]//2,:].get() #/ (data.shape[2]) # /data.shape[1] #data.shape[1]//2

@jax.jit
def dataDHTflip(data):
    """
    
        _Discrete Hartley Transform_
        This function implements the DHT based on the Real and Imaginary parts of the DFT.
        Here we flip the input signal so we can convolve the input and kernel using only 1
        pointwise product. We can only do that if we have an even signal being convolved. 
        
    """    
    # data now has shape (16, 4, 1000, 6)    
    dataflip = jnp.flip(data, axis=2)
    symdata = jnp.concatenate((dataflip, data), axis=2)   
    fourier_data = fft(symdata, axis=2)
    hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_hartley = jnp.abs(hartley_data)
    
    return norm_hartley



@jax.jit
def dataDHT_half(data):
    """
    
        _Discrete Hartley Transform_
        This function implements the DHT based on the Real and Imaginary parts of the DFT.
        Here we get only half the DHT based on its symmetry.
        
    """    
    # data now has shape (16, 4, 1000, 6)    
    fourier_data = fft(data, axis=2)
    hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_hartley = jnp.abs(hartley_data)
    
    return norm_hartley.at[:,:,:data.shape[2]//2,:].get() #/ (data.shape[2]) # /data.shape[1] #data.shape[1]//2

@jax.jit
def dataDHT_halfsym(data):
    """
    
        _Discrete Hartley Transform_
        This function implements the DHT based on the Real and Imaginary parts of the DFT.
        The input data is now symmetric. Here we get only half the DHT based on its symmetry.
        We also apply the hyperbolic arcsin as in the 2D scenario.
        NAME: 'symDHT'
    """    
    # data now has shape (16, 4, 1000, 6) 
    dataflip = jnp.flip(data, axis=2)
    symdata = jnp.concatenate((dataflip, data), axis=2)   
    fourier_data = fft(symdata, axis=2)
    hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_hartley = jnp.abs(jnp.arcsinh(hartley_data))
    
    return norm_hartley.at[:,:,:data.shape[2]//2,:].get()


@jax.jit
def dataDHT_halfsym_1(data):
    """
    
        _Discrete Hartley Transform_
        This function implements the DHT based on the Real and Imaginary parts of the DFT.
        The input data is now symmetric. Here we get only half the DHT based on its symmetry.
        We do NOT apply the hyperbolic arcsin as in the 2D scenario. 
        NAME: 'symDHTabs'
        
    """    
    # data now has shape (16, 4, 1000, 6) 
    dataflip = jnp.flip(data, axis=2)
    symdata = jnp.concatenate((dataflip, data), axis=2)   
    fourier_data = fft(symdata, axis=2)
    hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_hartley = jnp.abs(hartley_data)
    
    return norm_hartley.at[:,:,:data.shape[2]//2,:].get()


@jax.jit
def dataDHT_half_arcsinh(data):
    """
    
        _Discrete Hartley Transform_
        This function implements the DHT based on the Real and Imaginary parts of the DFT.
        Here we get only half the DHT based on its symmetry.
        
    """    
    # data now has shape (16, 4, 1000, 6)    
    fourier_data = fft(data, axis=2)
    hartley_data = jnp.real(fourier_data) - jnp.imag(fourier_data)

    # implement normalization if needed
    norm_hartley = jnp.arcsinh(hartley_data)
    
    return norm_hartley.at[:,:,:data.shape[2]//2,:].get() #/ (data.shape[2]) # /data.shape[1] #data.shape[1]//2