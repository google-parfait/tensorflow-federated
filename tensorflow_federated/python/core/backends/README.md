# Integration With Custom Backends

Computations expressed in TFF can be executed on a variety of backends,
including native backends that implement TFF's interfaces such as
`tff.framework.Executor`, as well as custom non-native backends such as
MapReduce-like systems that may only be able to run a subset of computations
expressibe in TFF. This directory contains utility classes dedicated for
interfacing such backends, a separate directory for each backend class.

The code in this directory can depend on the main TFF API (`core/api`), and on
`tff.framework`. It should not depend on the implementation.
