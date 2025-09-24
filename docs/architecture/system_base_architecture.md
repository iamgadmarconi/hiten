# System Module Architecture

## Module Mission

- Expose high-level CR3BP system primitives for end users
- Hide numerical concerns behind adapters into `hiten.algorithms`
- Provide a clean entry point for building, serialising, and exploring systems

## Design Principles

- **Separation of concerns:** Domain objects stay thin; numerics remain inside `algorithms/`
- **Immutable surface:** Public API favours immutable dataclasses and read-only properties
- **Adapter pattern:** Explicit adapter layer translates user requests into algorithm calls
- **Lazy orchestration:** Expensive resources (dynamical systems, tables) are created on demand and cached centrally
- **Friendly errors:** System layer normalises exceptions and documents expected usage

## File-Level Layout (`system/base.py`)

|       Section      |                         Responsibility                                   |
|      ---           |      ---                                                                 |
| Public API exports | `System`, helper factories, convenience functions                        |
| Configuration DTOs | `SystemConfig`, `SystemSnapshot` with validated metadata only            |
| Adapters           | `_SystemDynamicsAdapter`, `_SystemPersistenceAdapter`, `_SystemRegistry` |
| Services           | `_LibrationPointRegistry`, `_DynamicsRegistry`, `_OrbitBuilder`          |
| Internal utilities | Validation helpers, error translators, type aliases                      |

## Key Types

### `System`

- Immutable user-facing descriptor of a CR3BP instance
- Provides cached properties (primary, secondary, distance, μ)
- Delegates heavy operations to adapters via injected context
- Offers ergonomic helpers: `propagate`, `get_libration_point`
- Returns domain DTOs (`Trajectory`, `LibrationPointHandle`) instead of raw arrays

### `SystemContext`

- Lightweight container shared across child objects (libration points, orbits)
- Holds references to adapters/registries
- Ensures consistent access without leaking algorithm internals

### `SystemConfig`

- Pure metadata for constructing a `System`
- Stores identifiers (body names), numerical parameters, optional overrides
- Validates user input before handing off to adapters

### Adapter Classes

- `_SystemDynamicsAdapter`: resolves algorithm engines (propagation, STM) and executes calls
- `_SystemPersistenceAdapter`: bridges to `utils.io` loaders/savers
- `_SystemRegistry`: central cache for created adapters, preventing duplicate instantiation
- All adapters live in private scope; they never appear in the public API

## Dependency Flow

```bash
user -> system.base.System -> adapters -> algorithms.*
                              |
                          utils.io
```

- `system/` never imports algorithm internals at the top level; adapters encapsulate that knowledge
- Downstream system modules (`libration`, `orbits`, `manifold`) consume the `SystemContext` to request services instead of touching algorithms directly

## Public API Surface (Examples)

```python
system = System.from_bodies("earth", "moon")
trajectory = system.propagate(initial_state, duration=2 * np.pi)
L1 = system.get_libration_point(1)
family = L1.create_orbit("halo", amplitude=0.03)
system.save("earth_moon_system.h5")
```

- Each method validates inputs, delegates to an adapter, then wraps results in typed value objects

## Error Handling Strategy

- Convert algorithm exceptions into user-friendly `SystemError`, `PropagationError`, etc.
- Document failure modes (invalid bodies, integration divergence) in docstrings
- Surface actionable hints (e.g., "increase steps" or "adjust amplitude")

## Testing Guidelines

- Unit tests mock adapters to ensure public API behaviour
- Integration tests verify end-to-end workflow with real algorithms (opt-in, slower)
- Serialization round-trips confirm DTO fidelity without invoking numerics

## Implementation Checklist

- [ ] Introduce `SystemConfig`, `SystemContext`, adapter classes
- [ ] Refactor `System` to depend on adapters instead of algorithms directly
- [ ] Update downstream modules to pull services from `SystemContext`
- [ ] Provide comprehensive docstrings and usage examples
- [ ] Add unit tests covering new architecture contracts
