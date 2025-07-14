# HITEN - Computational Toolkit for the Circular Restricted Three-Body Problem

[![PyPI version](https://img.shields.io/pypi/v/hiten.svg?color=brightgreen)](https://pypi.org/project/hiten/)

## Overview

**HITEN** is a research-oriented Python library that provides an extensible implementation of high-order analytical and numerical techniques for the circular restricted three-body problem (CR3BP).

## Examples

1. **Parameterisation of periodic orbits and their invariant manifolds**

   The toolkit constructs periodic solutions such as halo orbits and computes their stable and unstable manifolds.

   ```python
   from hiten import System

   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)

   orbit = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   orbit.correct(max_attempts=25)
   orbit.propagate(steps=1000)

   manifold = orbit.manifold(stable=True, direction="positive")
   manifold.compute()
   manifold.plot()
   ```

   ![Halo orbit stable manifold](results/plots/halo_stable_manifold.svg)

   *Figure&nbsp;1 - Stable manifold of an Earth-Moon \(L_1\) halo orbit.*

   Knowing the dynamics of the center manifold, initial conditions for vertical orbits can be computed and associated manifolds created. These reveal natural transport channels that can be exploited for low-energy mission design.

   ```python
   from hiten import System, VerticalOrbit

   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)

   cm = l1.get_center_manifold(max_degree=10)
   cm.compute()

   initial_state = cm.ic(poincare_point=[0.0, 0.0], energy=0.6, section_coord="q3")

   orbit = VerticalOrbit(l1, initial_state=initial_state)
   orbit.correct(max_attempts=100)
   orbit.propagate(steps=1000)

   manifold = orbit.manifold(stable=True, direction="positive")
   manifold.compute()
   manifold.plot()
   ```

   ![Vertical orbit stable manifold](results/plots/vl_stable_manifold.svg)

   *Figure&nbsp;2 - Stable manifold of an Earth-Moon \(L_1\) vertical orbit.*

2. **Generating families of periodic orbits**

   The toolkit can generate families of periodic orbits by continuation.

   ```python
   from hiten import System
   from hiten.algorithms import StateParameter

    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)

    seed = l1.create_orbit('lyapunov', amplitude_x= 1e-3)
    seed.correct(max_attempts=25)

    target_amp = 1e-2 # grow A_x from 0.001 to 0.01 (relative amplitude)
    current_amp = seed.amplitude
    num_orbits = 10

    # Step in amplitude space (predictor still tweaks X component)
    step = (target_amp - current_amp) / (num_orbits - 1)

    engine = StateParameter(
        initial_orbit=seed,
        state=(S.X),     # underlying coordinate that gets nudged
        amplitude=True,  # but the continuation parameter is A_x
        target=(current_amp, target_amp),
        step=step,
        corrector_kwargs=dict(max_attempts=50, tol=1e-13),
        max_orbits=num_orbits,
    )
    engine.run()

    family = OrbitFamily.from_engine(engine)
    family.propagate()
    family.plot()
    ```

    ![Lyapunov orbit family](results/plots/lyapunov_family.svg)

    *Figure&nbsp;3 - Family of Earth-Moon \(L_1\) Lyapunov orbits.*

3. **Generating Poincaré maps**

   The toolkit can generate Poincaré maps for the centre manifold over various sections.

   ```python
   from hiten import System

   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)

   cm = l1.get_center_manifold(max_degree=12)
   cm.compute()

   pm = cm.poincare_map(energy=0.7, section_coord="q2", n_seeds=50, n_iter=100, seed_strategy="axis_aligned")
   pm.compute()
   pm.plot()
   ```

   ![Poincaré map](results/plots/poincare_map.svg)

   *Figure&nbsp;4 - Poincaré map of the centre manifold of the Earth-Moon \(L_1\) libration point using the \(q_2=0\) section.*

4. **Generating invariant tori**

   Hiten can generate invariant tori for periodic orbits.

   ```python
   from hiten import System
   from hiten.algorithms import InvariantTori

    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)

    orbit = l1.create_orbit('halo', amplitude_z=0.3, zenith='southern')
    orbit.correct(max_attempts=25)
    orbit.propagate(steps=1000)
   
    torus = InvariantTori(orbit)
    torus.compute(scheme='linear', epsilon=1e-2, n_theta1=256, n_theta2=256)
    torus.plot()
   ```

   ![Invariant tori](results/plots/invariant_tori.svg)

   *Figure&nbsp;5 - Invariant torus of an Earth-Moon \(L_1\) quasi-halo orbit.*
