import jax.numpy as jnp

def NormalizeData(X, min=0, max=1):
    
    ### data normalization between 0-1
    
    X_std = (X - jnp.min(X)) / (jnp.max(X) - jnp.min(X))
    X_scaled = X_std * (max - min) + min
    
    return X_scaled