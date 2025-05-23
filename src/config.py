
# System configuration
SYSTEM = "SE"  # "EM" for Earth-Moon or "SE" for Sun-Earth
L_POINT = 1    # Libration point number (1 or 2)

# Algorithm parameters
MAX_DEG = 5
TOL     = 1e-14

FASTMATH = False  # Global flag for Numba's fastmath option 

H0_LEVELS = [0.6] # [0.20, 0.40, 0.60, 1.00]
DT = 1e-2
USE_SYMPLECTIC = False
N_SEEDS = 1 # seeds along q2-axis
N_ITER = 100 # iterations per seed