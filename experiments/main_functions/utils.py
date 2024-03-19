import os

import pandas as pd
import jax
import jax.numpy as jnp
import subprocess
from flax import linen as nn

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

def my_bias_init(key, shape, dtype=jnp.float32):
    init = nn.initializers.glorot_normal()(key, (1,shape[0]), dtype)
    return jnp.ravel(init) 

def read_grid_search(path_to_results):
    """read_grid_search
    """
    #Create path to folder of interest
    folder_path = os.path.join(os.getcwd(),path_to_results)
    
    #create empty dataframe
    data = pd.DataFrame()
    
    for archive in os.listdir(folder_path):
        file = os.path.join(folder_path,archive)
        single_data = pd.read_csv(file, index_col=0)
        data = pd.concat([data,single_data],axis=0)
    
    return data.reset_index(drop=True)