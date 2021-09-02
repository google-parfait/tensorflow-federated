# Deployment

In addition to defining computations, TFF provides tools for executing them.
Whereas the primary focus is on simulations, the interfaces and tools we provide
are more general. This document outlines the options for deployment to various
types of platform.

Note: This document is still under construction.

## Overview

There are two principal modes of deployment for TFF computations:

*   **Native backends**. We're going to refer to a backend as *native* if it is
    capable of interpreting the syntactic structure of TFF computations as
    defined in
    [`computation.proto`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto).
    A native backend does not necessarily have to support all language
    constructs or intrinsics. Native backends must implement one of the standard
    TFF *executor* interfaces, such as
    [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor)
    for consumption by Python code, or the language-independent version of it
    defined in
    [`executor.proto`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/executor.proto)
    exposed as a gRPC endpoint.

    Native backends that support the above interfaces can be used interactively
    in lieu of the default reference runtime, e.g., to run notebooks or
    experiment scripts. Most native backends will operate in the *interpreted
    mode*, i.e., they will process the computation definition as it is defined,
    and execute it incrementally, but this does not always have to be the case.
    A native backend can also *transform* (*compile*, or JIT-compile) a part of
    the computation for better performance, or to simplify its structure. One
    example common use of this would be to reduce the set of federated operators
    that appear in a computation, so that parts of the backend dowstream of the
    transformation do not have to be exposed to the full set.

*   **Non-native backends**. Non-native backends, in contrast to the native
    ones, cannot directly interpret the TFF computation structure, and require
    it to be converted into a different *target representation* understood by
    the backend. A notable example of such a backend would be a Hadoop cluster,
    or a similar platform for static data pipelines. In order for a computation
    to be deployed to such a backend, it must first be *transformed* (or
    *compiled*). Depending on the setup, this can be done transparently to the
    user (i.e., a non-native backend could be wrapped in a standard executor
    interface such as
    [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor)
    that performs transformations under the hood), or it can be exposed as a
    tool that allows the user to manually convert a computation, or a set of
    computations, into the appropriate target representation understood by the
    particular class of backends. Code that supports specific types of
    non-native backends can be found in the
    [`tff.backends`](https://www.tensorflow.org/federated/api_docs/python/tff/backends)
    namespace. At the time of this writing, the only support type of non-native
    backends is a class of systems capable of executing single-round MapReduce.

## Native Backends

More details coming soon.

## Non-Native Backends

### MapReduce

More details coming soon.
