import jax
import jax.numpy as jnp
import subprocess

def NormalizeData(X, min=0, max=1):
    
    """
    Converts data to the interval 0-1.

    Args:
    - X: Input array data of shape (examples, attributes).
    - min: Minimum value of normalization.
    - max: Maximum value of normalization.
    
    Returns:
    - Array of the same dimension as X normalized between min and max values.
    """
    
    X_std = (X - jnp.min(X)) / (jnp.max(X) - jnp.min(X))
    X_scaled = X_std * (max - min) + min
    
    return X_scaled

def to_categorical(labels, n_classes=int):
    """
    Converts a class vector (integers) to binary class matrix.

    Args:
    - y: Array of integers representing classes.
    - n_classes: Total number of classes.

    Returns:
    - Array of one-hot encoded vectors.
    """
    y_one_hot = jnp.eye(n_classes)[labels]
    return y_one_hot

def get_gpu_memory_info():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip().split('\n')

    for line in output:
        free_memory, used_memory = map(int, line.split(','))
        
        print(f"Free GPU Memory: {free_memory} MiB")
        print(f"Used GPU Memory: {used_memory} MiB")

def my_init(key, shape, dtype=jnp.float32, mean=2.0, std=0.01):
    return mean + std * jax.random.normal(key, shape, dtype=dtype)