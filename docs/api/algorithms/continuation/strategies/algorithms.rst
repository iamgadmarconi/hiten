Continuation Algorithm Strategies
==================================

The strategies module has been restructured. Algorithm strategies are now implemented as stepping strategies within the main continuation module. See the stepping module documentation for details on available stepping strategies.

The stepping strategies provide the core algorithmic components for different continuation approaches:

**Natural Parameter Stepping**
    Simple stepping strategy for natural parameter continuation

**Secant Stepping**
    Pseudo-arclength continuation using secant-based tangent estimation

**Plain Stepping**
    Basic stepping strategy using user-provided predictor functions

These strategies are located in the main continuation stepping module rather than in a separate strategies submodule.
