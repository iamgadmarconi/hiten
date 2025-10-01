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
orbit.correct()
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
   orbit.correct()
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

   cm = l1.get_center_manifold(degree=6)
   cm.compute()

   initial_state = cm.to_synodic([0.0, 0.0], 0.6, "q3")

   orbit = VerticalOrbit(l1, initial_state=initial_state)
   orbit.correct()
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
   from hiten.algorithms.continuation.options import OrbitContinuationOptions
   from hiten.algorithms.types.states import SynodicState
   from hiten.system.family import OrbitFamily

   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   halo_seed = l1.create_orbit('halo', amplitude_z=0.2, zenith='southern')
   halo_seed.correct()
   halo_seed.propagate()

   options = OrbitContinuationOptions(
      target=(
         [halo_seed.initial_state[SynodicState.Z]], 
         [halo_seed.initial_state[SynodicState.Z] + 2.0]
      ),
      step=((0.02,),),
      max_members=100,
   )

   result = halo_seed.generate(options)

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

   cm = l1.get_center_manifold(degree=6)
   cm.compute()

   pm = cm.poincare_map(energy=0.7)
   pm.compute(section_coord="p3")
   pm.plot(axes=("p2", "q3"))
   ```

   ![Poincare map](results/plots/poincare_map.svg)

   *Figure&nbsp;4 - Poincare map of the centre manifold of the Earth-Moon \(L_1\) libration point using the \(q_2=0\) section.*

   Or the synodic section of a vertical orbit manifold:

   ```python
   from hiten import System, SynodicMap, VerticalOrbit

   system = System.from_bodies("earth", "moon")
   l_point = system.get_libration_point(1)

   cm = l_point.get_center_manifold(degree=6)
   cm.compute()

   ic_seed = cm.to_synodic([0.0, 0.0], 0.6, "q3") # Good initial guess from CM

   orbit = VerticalOrbit(l_point, initial_state=ic_seed)
   orbit.correct()
   orbit.propagate(steps=1000)

   manifold = orbit.manifold(stable=True, direction="positive")
   manifold.compute(step=0.005)
   manifold.plot()

   synodic_map = SynodicMap(manifold)
   synodic_map.compute(
      section_axis="y",
      section_offset=0.0,
      plane_coords=("x", "z"),
      direction=-1
   )
   synodic_map.plot()
   ```

   ![Synodic map](results/plots/synodic_map.svg)

   *Figure&nbsp;5 - Synodic map of the stable manifold of an Earth-Moon \(L_1\) vertical orbit.*

4. **Detecting heteroclinic connections**

   The toolkit can detect heteroclinic connections between two manifolds.

   ```python
    from hiten import System
    from hiten.algorithms.connections import ConnectionPipeline
    from hiten.algorithms.connections.config import ConnectionConfig
    from hiten.algorithms.connections.options import ConnectionOptions
    from hiten.algorithms.poincare import SynodicMapConfig

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
    )

    config = ConnectionConfig(
        section=section_cfg,
        direction=-1,
    )

    options = ConnectionOptions(
        delta_v_tol=1,
        ballistic_tol=1e-8,
        eps2d=1e-3,
    )
    
    conn = ConnectionPipeline.with_default_engine(config=config)

    result = conn.solve(manifold_l1, manifold_l2, options=options)

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
   from hiten import System, InvariantTori

    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)

    orbit = l1.create_orbit('halo', amplitude_z=0.3, zenith='southern')
    orbit.correct()
    orbit.propagate(steps=1000)
   
    torus = InvariantTori(orbit)
    torus.compute(epsilon=1e-2, n_theta1=512, n_theta2=512)
    torus.plot()
   ```

   ![Invariant tori](results/plots/invariant_tori.svg)

   *Figure&nbsp;7 - Invariant torus of an Earth-Moon \(L_1\) quasi-halo orbit.*

## Run the examples

Example scripts are in the `examples` directory. From the project root:

```powershell
py -m pip install -e .
python examples\periodic_orbits.py
python examples\orbit_manifold.py
python examples\orbit_family.py
python examples\poincare_map.py
python examples\synodic_map.py
python examples\heteroclinic_connection.py
python examples\invariant_tori.py
python examples\center_manifold.py
```

## Contributing

Issues and pull requests are welcome. For local development:

```powershell
py -m pip install -e .[dev]
python -m pytest -q
```

## License

This project is licensed under the terms of the MIT License. See `LICENSE`.
