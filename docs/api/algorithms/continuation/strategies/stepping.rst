Continuation Stepping Strategies
=================================

The stepping module provides concrete implementations of stepping strategies used in continuation algorithms. These strategies handle the prediction phase of the continuation process, generating numerical representations of the next solution based on the current solution and step size.

The stepping strategies have been moved to the main continuation stepping module. See the stepping module documentation for details on available stepping strategies.

**Extended Step Strategy**
    Defines an extended protocol for stepping strategies with event hooks

**Natural Parameter Stepping**
    Implements a natural parameter stepping strategy with user-supplied predictor

**Secant Stepping**
    Implements a secant-based stepping strategy for pseudo-arclength continuation

These strategies are now located in the main continuation stepping module rather than in a separate strategies submodule.
