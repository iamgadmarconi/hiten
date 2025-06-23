# HITEN - Computational Toolkit for the Circular Restricted Three-Body Problem

[![PyPI version](https://img.shields.io/pypi/v/hiten.svg?color=brightgreen)](https://pypi.org/project/hiten/)

## Overview

**HITEN** is a research-oriented Python library that provides an extensible implementation of high-order analytical and numerical techniques for the circular restricted three-body problem (CR3BP).

## Examples

1. **High-order parameterisation of periodic orbits and their invariant manifolds**

   The toolkit constructs periodic solutions such as halo orbits and obtains polynomial representations of their stable and unstable manifolds. This enables fast, mesh-free evaluation of trajectories seeded on these structures.

   ```python
   from hiten import System, HaloOrbit, Manifold

   system = System.from_bodies("earth", "moon")
   libration_point = system.get_libration_point(1)

   orbit = HaloOrbit(libration_point, Az=0.2, Zenith="southern")
   orbit.differential_correction(max_attempts=25)
   orbit.propagate(steps=1000)

   manifold = orbit.manifold(stable=True, direction="Positive")
   manifold.compute()
   manifold.plot()
   ```

   ![Halo orbit stable manifold](results/plots/halo_stable_manifold.svg)

   *Figure&nbsp;1 – Stable manifold emanating from a halo orbit around the \(L_1\) libration point.*

2. **Computation of Lyapunov families and associated transport pathways**

   Built-in continuation routines retrieve vertical Lyapunov orbits of varying amplitudes. Their invariant manifolds reveal natural transport channels that can be exploited for low-energy mission design.

   ```python
   from hiten import System, VerticalLyapunovOrbit, CenterManifold, Manifold

   system = System.from_bodies("earth", "moon")
   libration_point = system.get_libration_point(1)

   cm = CenterManifold(libration_point, 10)
   cm.compute()

   initial_state = cm.ic([0.0, 0.0], 0.6, "q3")

   orbit = VerticalLyapunovOrbit(libration_point, initial_state=initial_state)
   orbit.differential_correction(max_attempts=100)
   orbit.propagate(steps=1000)

   manifold = orbit.manifold(stable=True, direction="Positive")
   manifold.compute()
   manifold.plot()
   ```

   ![Vertical Lyapunov orbit stable manifold](results/plots/vl_stable_manifold.svg)

   *Figure&nbsp;2 – Stable manifold corresponding to a vertical Lyapunov orbit.*
