import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import subprocess
from flax import linen as nn
from sklearn.preprocessing import MinMaxScaler

def shuffling(x_test, y_test):
    k = np.random.randint(1000)
    idx_test = jax.random.permutation(jax.random.PRNGKey(k), x_test.shape[0])
    return x_test[idx_test], y_test[idx_test]

def NormalizeData(X, min=0, max=1, axis=2):
    # not using yet
    """
    Converts data to the interval 0-1.

    Args:
    - X: Input array data of shape (examples, attributes).
    - min: Minimum value of normalization.
    - max: Maximum value of normalization.
    
    Returns:
    - Array of the same dimension as X normalized between min and max values.
    """
    # data now has shape (16, 4, 1000, 6) >>>>>>> (144, 13*500)
    min_vals = jnp.min(X,axis=axis, keepdims=True)
    max_vals = jnp.max(X, axis=axis, keepdims=True)
    X_std = jnp.divide(X-min_vals,max_vals - min_vals)
    X_scaled = X_std * (max-min) + min
    #X_std = (X - jnp.min(X,axis=axis).reshape(X.shape[0],X.shape[1],1,X.shape[-1])) / (jnp.max(X,axis=2).reshape(X.shape[0],X.shape[1],1,X.shape[-1]) - jnp.min(X,axis=2).reshape(X.shape[0],X.shape[1],1,X.shape[-1]))
    #X_std = X - jnp.min(X, axis=axis)
    #X_scaled = X_std * (max - min) + min
    #min_vals = np.min(X, axis=axis)
    #max_vals = np.max(X, axis=axis)
    #normalized_data = (X - min_vals) / (max_vals - min_vals)
    #return normalized_data
    return X_scaled

def NormalizeData_(data):
    # Reshape the data to make it 2D while preserving the other axes
    reshaped_data = data.reshape((-1, data.shape[2]))

    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to your data and transform it
    normalized_data = scaler.fit_transform(reshaped_data)

    # Reshape the normalized data back to its original shape
    normalized_data = normalized_data.reshape(data.shape)
    return normalized_data

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

def sel_trials(trials, val_trial: int):
    train_trials = jnp.delete(trials, jnp.where(trials == val_trial)[0][0])
    return train_trials

def count_params(dct):
    soma = 0
    for k,v in dct.items():
        if isinstance(v, dict):
            soma+=count_params(v)
        if isinstance(v, tuple):
            soma+=np.prod(v)
        if isinstance(v, int):
            soma+=v
    return soma