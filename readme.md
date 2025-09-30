![HITEN](results/plots/hiten-cropped.svg)

# HITEN - Computational Toolkit for the Circular Restricted Three-Body Problem

[![PyPI version](https://img.shields.io/pypi/v/hiten.svg?color=brightgreen)](https://pypi.org/project/hiten/)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://iamgadmarconi.github.io/hiten/)

## Overview

**HITEN** is a research-oriented Python library that provides an extensible implementation of high-order analytical and numerical techniques for the circular restricted three-body problem (CR3BP).

## Installation

HITEN is published on PyPI. A recent Python version (3.9+) is required.

```bash
py -m pip install hiten
```

## Quickstart

Full documentation is available [here](https://iamgadmarconi.github.io/hiten/).

Compute a halo orbit around Earth-Moon L1 and plot a branch of its stable manifold:

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

   cm = l1.get_center_manifold(degree=10)
   cm.compute()

   initial_state = cm.to_synodic(poincare_point=[0.0, 0.0], energy=0.6, section_coord="q3")

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
   from hiten import System, OrbitFamily
   from hiten.algorithms import ContinuationPipeline
   from hiten.algorithms.types.states import SynodicState
   from hiten.algorithms.continuation.config import _OrbitContinuationConfig

   num_orbits = 50
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   halo_seed = l1.create_orbit('halo', amplitude_z= 0.2, zenith='southern')
   halo_seed.correct(max_attempts=25, max_delta=1e-3)

   current_z = halo_seed.initial_state[SynodicState.Z]  # 0 for planar Lyapunov halo_seed
   target_z = current_z + 5.0   # introduce out-of-plane Z
   step_z = (target_z - current_z) / (num_orbits - 1)

   config= _OrbitContinuationConfig(
      target=([current_z], [target_z]),
      step=((step_z),),
      state=(SynodicState.Z,),
      max_members=50,
      extra_params=dict(max_attempts=50, tol=1e-12),
      stepper="secant",
   )

   state_parameter = ContinuationPipeline.with_default_engine(config=config)

   result = state_parameter.generate(halo_seed)

   logger.info(f"Generated {len(result.family)} orbits (success rate {result.success_rate:.2%})")

   family = OrbitFamily.from_result(result)
   family.propagate()
   family.plot()
   ```

    ![Halo orbit family](results/plots/halo_family.svg)

    *Figure&nbsp;3 - Family of Earth-Moon \(L_1\) Halo orbits.*

3. **Generating Poincare maps**

   The toolkit can generate Poincare maps for arbitrary sections. For example, the centre manifold of the Earth-Moon \(L_1\) libration point:

   ```python
   from hiten import System

   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)

   cm = l1.get_center_manifold(degree=12)
   cm.compute()

   pm = cm.poincare_map(energy=0.7, section_coord="q2", n_seeds=50, n_iter=100, seed_strategy="axis_aligned")
   pm.compute()
   pm.plot()
   ```

   ![Poincare map](results/plots/poincare_map.svg)

   *Figure&nbsp;4 - Poincare map of the centre manifold of the Earth-Moon \(L_1\) libration point using the \(q_2=0\) section.*

   Or the synodic section of a vertical orbit manifold:

   ```python
   from hiten import System, VerticalOrbit
   from hiten.algorithms import SynodicMap, SynodicMapConfig

   system = System.from_bodies("earth", "moon")
   l_point = system.get_libration_point(1)

   cm = l_point.get_center_manifold(degree=6)
   cm.compute()

   ic_seed = cm.to_synodic([0.0, 0.0], 0.6, "q3") # Good initial guess from CM

   orbit = VerticalOrbit(l_point, initial_state=ic_seed)
   orbit.correct(max_attempts=100, finite_difference=True)
   orbit.propagate(steps=1000)

   manifold = orbit.manifold(stable=True, direction="positive")
   manifold.compute(step=0.005)
   manifold.plot()

   section_cfg = SynodicMapConfig(
      section_axis="y",
      section_offset=0.0,
      plane_coords=("x", "z"),
      interp_kind="cubic",
      segment_refine=30,
      newton_max_iter=10,
   )
   synodic_map = SynodicMap(section_cfg)
   synodic_map.from_manifold(manifold)
   synodic_map.plot()
   ```

   ![Synodic map](results/plots/synodic_map.svg)

   *Figure&nbsp;5 - Synodic map of the stable manifold of an Earth-Moon \(L_1\) vertical orbit.*

4. **Detecting heteroclinic connections**

   The toolkit can detect heteroclinic connections between two manifolds.

   ```python
    system = System.from_bodies("earth", "moon")
    mu = system.mu

    l1 = system.get_libration_point(1)
    l2 = system.get_libration_point(2)

    halo_l1 = l1.create_orbit('halo', amplitude_z=0.5, zenith='southern')
    halo_l1.correct()
    halo_l1.propagate()

    halo_l2 = l2.create_orbit('halo', amplitude_z=0.3663368, zenith='northern')
    halo_l2.correct()
    halo_l2.propagate()

    manifold_l1 = halo_l1.manifold(stable=True, direction='positive')
    manifold_l1.compute(integration_fraction=0.9, step=0.005)

    manifold_l2 = halo_l2.manifold(stable=False, direction='negative')
    manifold_l2.compute(integration_fraction=1.0, step=0.005)

    section_cfg = SynodicMapConfig(
        section_axis="x",
        section_offset=1 - mu,
        plane_coords=("y", "z"),
        interp_kind="cubic",
        segment_refine=30,
        tol_on_surface=1e-9,
        dedup_time_tol=1e-9,
        dedup_point_tol=1e-9,
        max_hits_per_traj=None,
        n_workers=None,
    )

    # Create unified configuration with all parameters in one object
    config = _ConnectionConfig(
        section=section_cfg,        # Synodic section configuration
        direction=-1,                # Crossing direction (None = both directions)
        delta_v_tol=1,             # Maximum Delta-V tolerance
        ballistic_tol=1e-8,        # Threshold for ballistic classification
        eps2d=1e-3,                # 2D pairing radius
    )
    
    # Create connection using the factory method with unified config
    conn = ConnectionPipeline.with_default_engine(config=config)

    result = conn.solve(manifold_l1, manifold_l2)

    print(result)

    conn.plot(dark_mode=True)

    conn.plot_connection(dark_mode=True)
   ```

   ![Heteroclinic connection](results/plots/heteroclinic_connection.svg)

   *Figure&nbsp;6 - Heteroclinic connection between the stable manifold of an Earth-Moon \(L_1\) halo orbit and the unstable manifold of an Earth-Moon \(L_2\) halo orbit.*

   We can also plot the trajectories making up the connection:

   ![Heteroclinic connection trajectories](results/plots/heteroclinic_trajectory.svg)

   *Figure&nbsp;7 - One of the detected connections.*

5. **Generating invariant tori**

   Hiten can generate invariant tori for periodic orbits.

   ```python
   from hiten import System
   from hiten import InvariantTori

    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)

    orbit = l1.create_orbit('halo', amplitude_z=0.3, zenith='southern')
    orbit.correct(max_attempts=25)
    orbit.propagate(steps=1000)
   
    torus = InvariantTori(orbit)
    torus.compute(epsilon=1e-2, n_theta1=256, n_theta2=256)
    torus.plot()
   ```

   ![Invariant tori](results/plots/invariant_tori.svg)

   *Figure&nbsp;7 - Invariant torus of an Earth-Moon \(L_1\) quasi-halo orbit.*

## Run the examples

Example scripts are in the `examples` directory. From the project root:

```powershell
py -m pip install -e .
python examples\periodic_orbits.py
python examples\orbit_family.py
python examples\synodic_map.py
python examples\heteroclinic_connection.py
```

## Contributing

Issues and pull requests are welcome. For local development:

```powershell
py -m pip install -e .[dev]
python -m pytest -q
```

## License

This project is licensed under the terms of the MIT License. See `LICENSE`.
