import cProfile
import pstats
from main import main
import logging

# Configure logging to see the output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Run with profiler
print("Running with profile...")
cProfile.run('main()', filename='center_manifold.prof')

# Print profile results
print("\nProfile results:")
p = pstats.Stats('center_manifold.prof')
p.strip_dirs().sort_stats('cumulative').print_stats(30)  # Show top 30 functions by cumulative time

print("\nCalling functions:")
p.sort_stats('cumulative').print_callers(20)  # Show which functions call the top 20 time-consuming functions

print("\nCalled functions:")
p.sort_stats('cumulative').print_callees(20)  # Show which functions are called by the top 20 time-consuming functions 