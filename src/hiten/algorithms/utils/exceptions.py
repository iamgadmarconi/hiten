"""
Custom exceptions for the algorithms package.
"""

class HitenError(Exception):
    """Base exception for Hiten errors.
    
    Parameters
    ----------
    message : str
        The error message.
    """

    def __init__(self, message: str):
        super().__init__(message)


class ConvergenceError(HitenError):
    """Raised when an algorithm fails to converge.
    
    Parameters
    ----------
    message : str
        The error message.
    """

    def __init__(self, message: str):
        super().__init__(message)


class BackendError(HitenError):
    """An error occurred within a Backend algorithm.
    
    Parameters
    ----------
    message : str
        The error message.
    """

    def __init__(self, message: str):
        super().__init__(message)


class EngineError(HitenError):
    """An error occurred during an Engine's workflow.
    
    Parameters
    ----------
    message : str
        The error message.
    """

    def __init__(self, message: str):
        super().__init__(message)
