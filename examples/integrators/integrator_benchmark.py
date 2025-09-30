#!/usr/bin/env python3
"""Integrator benchmark script.

Tests Hiten integrators against SciPy on various ODE problems and provides
detailed accuracy, performance, and error analysis.
"""

import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', "src"))

from hiten.algorithms.dynamics.rhs import create_rhs_system
from hiten.algorithms.integrators import (AdaptiveRK, RungeKutta)

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class TestResult:
    """Store results from a single integrator test."""
    integrator_name: str
    problem_name: str
    final_error: float
    max_error: float
    relative_error: float
    computation_time: float
    n_steps: int
    energy_error: float = None
    converged: bool = True
    error_message: str = ""


# Test problems as simple functions
def harmonic_oscillator(t, y):
    """Harmonic oscillator: x'' + x = 0."""
    return np.array([y[1], -y[0]])

def van_der_pol(t, y):
    """Van der Pol oscillator: x'' - μ(1-x²)x' + x = 0."""
    mu = 1.0
    return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])

def lorenz(t, y):
    """Lorenz system: chaotic attractor."""
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    return np.array([
        sigma * (y[1] - y[0]),
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - beta * y[2]
    ])

def kepler(t, y):
    """Kepler problem: 2D central force motion."""
    x, y_pos, vx, vy = y
    mu = 1.0
    r = np.sqrt(x**2 + y_pos**2)
    r3 = r**3
    return np.array([vx, vy, -mu * x / r3, -mu * y_pos / r3])

def duffing(t, y):
    """Duffing oscillator: x'' + δx' + αx + βx³ = γcos(ωt)."""
    delta, alpha, beta = 0.1, -1.0, 1.0
    gamma, omega = 0.3, 1.2
    return np.array([
        y[1],
        -delta * y[1] - alpha * y[0] - beta * y[0]**3 + 
        gamma * np.cos(omega * t)
    ])

# Test problem configurations
TEST_PROBLEMS = [
    {
        "name": "Harmonic Oscillator",
        "rhs": harmonic_oscillator,
        "y0": np.array([1.0, 0.0]),
        "t_span": (0.0, 4*np.pi),
        "exact": lambda t: np.column_stack([np.cos(t), -np.sin(t)]),
        "energy": lambda y: 0.5 * (y[0]**2 + y[1]**2)
    },
    {
        "name": "Van der Pol",
        "rhs": van_der_pol,
        "y0": np.array([2.0, 0.0]),
        "t_span": (0.0, 20.0),
        "exact": None,
        "energy": None
    },
    {
        "name": "Lorenz",
        "rhs": lorenz,
        "y0": np.array([1.0, 1.0, 1.0]),
        "t_span": (0.0, 20.0),
        "exact": None,
        "energy": None
    },
    {
        "name": "Kepler",
        "rhs": kepler,
        "y0": np.array([1.0, 0.0, 0.0, 0.8]),
        "t_span": (0.0, 10.0),
        "exact": None,
        "energy": lambda y: 0.5 * (y[2]**2 + y[3]**2) - 1.0 / np.sqrt(y[0]**2 + y[1]**2)
    },
    {
        "name": "Duffing",
        "rhs": duffing,
        "y0": np.array([1.0, 0.0]),
        "t_span": (0.0, 20.0),
        "exact": None,
        "energy": None
    }
]


def run_integrator_test(integrator, system, problem_config, t_eval: np.ndarray) -> TestResult:
    """Run a single integrator test and return results."""
    
    # Warm-up JIT compilation (excluded from timing)
    try:
        if t_eval.size >= 2:
            dt = t_eval[1] - t_eval[0]
            warmup_t = np.array([t_eval[0], t_eval[0] + max(dt, 1e-8)], dtype=float)
        else:
            warmup_t = np.array([t_eval[0], t_eval[0] + 1e-8], dtype=float)
        _ = integrator.integrate(system, problem_config["y0"], warmup_t)
    except Exception:
        pass

    start_time = time.perf_counter()
    
    try:
        solution = integrator.integrate(system, problem_config["y0"], t_eval)
        computation_time = time.perf_counter() - start_time
        
        y_solution = solution.states
        
        # Calculate errors
        if problem_config["exact"] is not None:
            y_exact = problem_config["exact"](t_eval)
            errors = np.abs(y_solution - y_exact)
            final_error = np.linalg.norm(errors[-1])
            max_error = np.max(errors)
            relative_error = max_error / (np.max(np.abs(y_exact)) + 1e-16)
        else:
            # Use SciPy as reference
            scipy_sol = solve_ivp(
                system.rhs, problem_config["t_span"], problem_config["y0"], 
                t_eval=t_eval, rtol=1e-12, atol=1e-14
            )
            y_ref = scipy_sol.y.T
            errors = np.abs(y_solution - y_ref)
            final_error = np.linalg.norm(errors[-1])
            max_error = np.max(errors)
            relative_error = max_error / (np.max(np.abs(y_ref)) + 1e-16)
        
        # Calculate energy error if applicable
        energy_error = None
        if problem_config["energy"] is not None:
            initial_energy = problem_config["energy"](problem_config["y0"])
            final_energy = problem_config["energy"](y_solution[-1])
            energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
        
        # Count steps (approximate for fixed-step methods)
        n_steps = len(t_eval) - 1
        
        return TestResult(
            integrator_name=str(integrator),
            problem_name=problem_config["name"],
            final_error=final_error,
            max_error=max_error,
            relative_error=relative_error,
            computation_time=computation_time,
            n_steps=n_steps,
            energy_error=energy_error,
            converged=True
        )
        
    except Exception as e:
        computation_time = time.perf_counter() - start_time
        return TestResult(
            integrator_name=str(integrator),
            problem_name=problem_config["name"],
            final_error=np.inf,
            max_error=np.inf,
            relative_error=np.inf,
            computation_time=computation_time,
            n_steps=0,
            converged=False,
            error_message=str(e)
        )


def run_benchmark():
    """Run comprehensive tests on all integrators and problems."""
    
    # Define integrators to test
    integrators = {
        # Fixed-step RK methods
        "RK4": RungeKutta(order=4),
        "RK6": RungeKutta(order=6),
        "RK8": RungeKutta(order=8),
        
        # Adaptive RK methods
        "RK45": AdaptiveRK(order=5, rtol=1e-8, atol=1e-10),
        "DOP853": AdaptiveRK(order=8, rtol=1e-8, atol=1e-10),
    }
    
    # SciPy reference
    scipy_integrators = {
        "SciPy-RK45": "RK45",
        "SciPy-DOP853": "DOP853",
        "SciPy-RK23": "RK23",
        "SciPy-BDF": "BDF",
        "SciPy-LSODA": "LSODA"
    }
    
    # Time grid for evaluation
    t_eval = np.linspace(0, 10, 1001)
    
    all_results = []
    
    print("Running comprehensive integrator tests...")
    print("=" * 60)
    
    for problem_config in TEST_PROBLEMS:
        print(f"\nTesting problem: {problem_config['name']}")
        print("-" * 40)
        # Build a single dynamical system instance per problem for all tests
        system = create_rhs_system(problem_config["rhs"], dim=len(problem_config["y0"]), name=problem_config["name"])
        
        # Test Hiten integrators
        for name, integrator in integrators.items():
            result = run_integrator_test(integrator, system, problem_config, t_eval)
            all_results.append(result)
            
            if result.converged:
                print(f"{name:15s}: "
                      f"Final Error: {result.final_error:.2e}, "
                      f"Max Error: {result.max_error:.2e}, "
                      f"Time: {result.computation_time:.4f}s")
                
                if result.energy_error is not None:
                    print(f"{'':15s} Energy Error: {result.energy_error:.2e}")
            else:
                print(f"{name:15s}: FAILED - {result.error_message}")
        
        # Test SciPy integrators
        for name, method in scipy_integrators.items():
            start_time = time.perf_counter()
            
            try:
                scipy_sol = solve_ivp(
                    system.rhs, problem_config["t_span"], problem_config["y0"],
                    t_eval=t_eval, method=method, rtol=1e-8, atol=1e-10
                )
                computation_time = time.perf_counter() - start_time
                
                if scipy_sol.success:
                    y_solution = scipy_sol.y.T
                    
                    # Calculate errors
                    if problem_config["exact"] is not None:
                        y_exact = problem_config["exact"](t_eval)
                        errors = np.abs(y_solution - y_exact)
                        final_error = np.linalg.norm(errors[-1])
                        max_error = np.max(errors)
                        relative_error = max_error / (np.max(np.abs(y_exact)) + 1e-16)
                    else:
                        # Use highest accuracy SciPy as reference
                        ref_sol = solve_ivp(
                            system.rhs, problem_config["t_span"], problem_config["y0"],
                            t_eval=t_eval, method="DOP853", rtol=1e-12, atol=1e-14
                        )
                        y_ref = ref_sol.y.T
                        errors = np.abs(y_solution - y_ref)
                        final_error = np.linalg.norm(errors[-1])
                        max_error = np.max(errors)
                        relative_error = max_error / (np.max(np.abs(y_ref)) + 1e-16)
                    
                    # Energy error
                    energy_error = None
                    if problem_config["energy"] is not None:
                        initial_energy = problem_config["energy"](problem_config["y0"])
                        final_energy = problem_config["energy"](y_solution[-1])
                        energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
                    
                    result = TestResult(
                        integrator_name=name,
                        problem_name=problem_config["name"],
                        final_error=final_error,
                        max_error=max_error,
                        relative_error=relative_error,
                        computation_time=computation_time,
                        n_steps=len(scipy_sol.t) - 1,
                        energy_error=energy_error,
                        converged=True
                    )
                    all_results.append(result)
                    
                    print(f"{name:15s}: "
                          f"Final Error: {result.final_error:.2e}, "
                          f"Max Error: {result.max_error:.2e}, "
                          f"Time: {result.computation_time:.4f}s")
                    
                    if result.energy_error is not None:
                        print(f"{'':15s} Energy Error: {result.energy_error:.2e}")
                else:
                    print(f"{name:15s}: FAILED - {scipy_sol.message}")
                    
            except Exception as e:
                print(f"{name:15s}: ERROR - {e}")
    
    return all_results


def print_summary_table(results: List[TestResult]):
    """Print a summary table of all results."""
    
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Integrator':<15} {'Problem':<20} {'Final Error':<12} {'Max Error':<12} {'Time (s)':<10} {'Energy Error':<12}")
    print("-" * 100)
    
    # Group by problem for better readability
    problems = {}
    for result in results:
        if result.problem_name not in problems:
            problems[result.problem_name] = []
        problems[result.problem_name].append(result)
    
    for problem_name in sorted(problems.keys()):
        problem_results = problems[problem_name]
        for result in sorted(problem_results, key=lambda x: x.final_error):
            energy_str = f"{result.energy_error:.2e}" if result.energy_error is not None else "N/A"
            print(f"{result.integrator_name:<15} {result.problem_name:<20} "
                  f"{result.final_error:<12.2e} {result.max_error:<12.2e} "
                  f"{result.computation_time:<10.4f} {energy_str:<12}")


def create_performance_plots(results: List[TestResult]):
    """Create performance comparison plots."""
    
    # Create results directory
    os.makedirs('_debug/results/plots', exist_ok=True)
    
    # Extract data for plotting
    integrators = list(set(r.integrator_name for r in results if r.converged))
    problems = list(set(r.problem_name for r in results if r.converged))
    
    # Create accuracy vs speed plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(integrators)))
    integrator_colors = dict(zip(integrators, colors))
    
    # Plot 1: Final Error vs Time
    for integrator in integrators:
        integrator_results = [r for r in results if r.integrator_name == integrator and r.converged]
        if integrator_results:
            times = [r.computation_time for r in integrator_results]
            errors = [r.final_error for r in integrator_results]
            ax1.loglog(times, errors, 'o', color=integrator_colors[integrator], 
                      label=integrator, markersize=8)
    
    ax1.set_xlabel('Computation Time (s)')
    ax1.set_ylabel('Final Error')
    ax1.set_title('Accuracy vs Speed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max Error vs Time
    for integrator in integrators:
        integrator_results = [r for r in results if r.integrator_name == integrator and r.converged]
        if integrator_results:
            times = [r.computation_time for r in integrator_results]
            errors = [r.max_error for r in integrator_results]
            ax2.loglog(times, errors, 'o', color=integrator_colors[integrator], 
                      label=integrator, markersize=8)
    
    ax2.set_xlabel('Computation Time (s)')
    ax2.set_ylabel('Max Error')
    ax2.set_title('Max Error vs Speed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('_debug/results/plots/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Performance plots saved to _debug/results/plots/")


def main():
    """Main function to run all tests."""
    
    print("Hiten Integrators vs SciPy Comprehensive Benchmark")
    print("=" * 60)
    
    # Run comprehensive tests
    results = run_benchmark()
    
    # Print summary
    print_summary_table(results)
    
    # Create plots
    create_performance_plots(results)
    
    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()