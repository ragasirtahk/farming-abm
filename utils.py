import numpy as np

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    if len(array) == 0:
        return 0
    array = np.array(array, dtype=np.float64)  # Ensure floating-point type
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)  # Ensure all values are non-negative
    array += 0.0000001  # Avoid zero division
    array = np.sort(array)  # Sort array
    index = np.arange(1, array.shape[0] + 1)  # Index starts at 1
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))