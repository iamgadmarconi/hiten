Continuation Interface Protocols
=================================

The interfaces module defines protocol and base classes for continuation stepping strategies. These protocols specify the required interface for all stepping strategies used in continuation algorithms.

The base interfaces and protocols have been moved to the main continuation stepping module. See the stepping module documentation for details on the base interfaces and protocols.

**Base Step Interface**
    Defines the protocol for continuation stepping strategies

**Plain Stepping**
    Simple stepping strategy using a provided predictor function

These interfaces are now located in the main continuation stepping module rather than in a separate strategies submodule.
