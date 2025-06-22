# HITEN - Computational Toolkit for the Circular Restricted Three-Body Problem

## Overview

**HITEN** is a research-oriented Python library that provides an extensible implementation of high-order analytical and numerical techniques for the circular restricted three-body problem (CR3BP).

## Examples

1. **High-order parameterisation of periodic orbits and their invariant manifolds**

   The toolkit constructs periodic solutions such as halo orbits and obtains polynomial representations of their stable and unstable manifolds. This enables fast, mesh-free evaluation of trajectories seeded on these structures.

   ![Halo orbit stable manifold](results/plots/halo_stable_manifold.svg)

   *Figure&nbsp;1 – Stable manifold emanating from a halo orbit around the \(L_1\) libration point.*

2. **Computation of Lyapunov families and associated transport pathways**

   Built-in continuation routines retrieve vertical Lyapunov orbits of varying amplitudes. Their invariant manifolds reveal natural transport channels that can be exploited for low-energy mission design.

   ![Vertical Lyapunov orbit stable manifold](results/plots/vl_stable_manifold.svg)

   *Figure&nbsp;2 – Stable manifold corresponding to a vertical Lyapunov orbit.*
