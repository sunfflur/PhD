import jax.numpy as jnp

def windowing(data, labels, window, overlap):
    """
    Function to perform windowing on a signal.

    Args:
    - signal: The input signal (1D array).
    - window_size: Size of the window in samples.
    - overlap: Overlap between consecutive windows in samples.

    Returns:
    - windows: A 2D array containing the windowed segments of the signal.
    """
    
    N = data.shape[2]
    sampling_frequency = 250 #Hz
    t = jnp.arange(0,N) / sampling_frequency # 0 to 6 s
   
    num_windows = (N - (window*sampling_frequency)) // (window*sampling_frequency - overlap*sampling_frequency) + 1
    windows = []
    labels_samples = []
    for i in range(num_windows):
        start = i * (window*sampling_frequency - overlap*sampling_frequency)
        end = start + window*sampling_frequency
        sample = data[:, :, start:end, :]
        windows.append(sample)
        labels_samples.append(labels)
    samples = jnp.concatenate(windows, axis=1)
    nlabels = jnp.concatenate(labels_samples, axis=1)
    return samples, nlabels