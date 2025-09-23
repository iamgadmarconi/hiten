import numpy as np


def _totime(t, tf):
    """Find indices of closest time values in array.

    Searches time array for indices where values are closest to specified
    target times. Useful for extracting trajectory points at specific times.

    Parameters
    ----------
    t : array_like
        Time array to search.
    tf : float or array_like
        Target time value(s) to locate.

    Returns
    -------
    ndarray
        Indices where t values are closest to corresponding tf values.

    Notes
    -----
    - Uses absolute time values, so signs are ignored
    - Particularly useful for periodic orbit analysis
    - Returns single index for scalar tf, array of indices for array tf
    
    Examples
    --------
    >>> import numpy as np
    >>> from hiten.algorithms.common.mani import _totime
    >>> t = np.linspace(0, 10, 101)  # Time array
    >>> tf = [2.5, 7.1]  # Target times
    >>> indices = _totime(t, tf)
    >>> t[indices]  # Closest actual times
    array([2.5, 7.1])
    """
    # Convert to absolute values and ensure tf is array
    t = np.abs(t)
    tf = np.atleast_1d(tf)
    
    # Find closest indices
    I = np.empty(tf.shape, dtype=int)
    for k, target in enumerate(tf):
        diff = np.abs(target - t)
        I[k] = np.argmin(diff)
    
    return I