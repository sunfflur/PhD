import jax.numpy as jnp

def windowing(data, sampling_frequency,  window, overlap):
    """
    Function to perform windowing on a signal.

    Args:
    - signal: The input signal (1D array).
    - window_size: Size of the window in seconds.
    - overlap: Overlap between consecutive windows in seconds.

    Returns:
    - windows: A 2D array containing the windowed segments of the signal.
    """
    
    N = data.shape[2]
    #sampling_frequency = sampling_frequency #Hz
    #t = jnp.arange(0,N) / sampling_frequency # 0 to 6 s
   
    num_windows = int((N - (window*sampling_frequency)) // (window*sampling_frequency - overlap*sampling_frequency) + 1)
    print(num_windows)
    windows = []

    for i in range(num_windows): # janelas
        start = i * (window*sampling_frequency - overlap*sampling_frequency)
        end = start + window*sampling_frequency
        print(start, end)
        sample = data[:, :, int(start):int(end), :]
        windows.append(sample)
            
    samples = jnp.concatenate(windows, axis=3)
            
    return samples