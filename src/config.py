PRIMARY = "Earth"
SECONDARY = "Moon"
L_POINT = 1    # Libration point number (1 or 2)

# Algorithm parameters
MAX_DEG = 12
TOL = 1e-14

FASTMATH = False  # Global flag for Numba's fastmath option 

H0_LEVELS = [0.6] # [0.20, 0.40, 0.60, 1.00]
DT = 1e-2
USE_SYMPLECTIC = False
INTEGRATOR_ORDER = 6
C_OMEGA_HEURISTIC = 100.0
N_SEEDS = 1 # seeds along q2-axis
N_ITER = 15 # iterations per seed

# Precision control
USE_ARBITRARY_PRECISION = True  # Set to True to enable mpmath for critical computations
MPMATH_DPS = 50  # Decimal places for mpmath (default 50, standard float64 ≈ 15-17)
NUMPY_DTYPE_REAL = "float64"  # "float32" or "float64" 
NUMPY_DTYPE_COMPLEX = "complex128"  # "complex64" or "complex128"

# GPU configuration
USE_GPU = False
N_VARS = 6