import jax.numpy as jnp

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

