# System configuration
SYSTEM = "SE"  # "EM" for Earth-Moon or "SE" for Sun-Earth
L_POINT = 1    # Libration point number (1 or 2)

# Algorithm parameters
MAX_DEG = 12
TOL = 1e-14

FASTMATH = False  # Global flag for Numba's fastmath option 

# Precision control
USE_ARBITRARY_PRECISION = False  # Set to True to enable mpmath for critical computations
MPMATH_DPS = 50  # Decimal places for mpmath (default 50, standard float64 ≈ 15-17)
NUMPY_DTYPE_REAL = "float64"  # "float32" or "float64" 
NUMPY_DTYPE_COMPLEX = "complex128"  # "complex64" or "complex128"

H0_LEVELS = [0.6] # [0.20, 0.40, 0.60, 1.00]
DT = 1e-3
USE_SYMPLECTIC = True
N_SEEDS = 1 # seeds along q2-axis
N_ITER = 100 # iterations per seed