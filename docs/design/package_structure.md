# Package Structure

[TOC]

## Overview

### Terminology

#### Python Module

A Python module is a file containing Python definitions and statements. See
[modules](https://docs.python.org/3/tutorial/modules.html#modules) for more
information.

#### Python Package

Python packages are a way of structuring Python modules. See
[packages](https://docs.python.org/3/tutorial/modules.html#packages) for more
information.

#### Public TFF API

The TFF API that is exposed by the
[TFF API documentation](https://www.tensorflow.org/federated/api_docs/python/tff);
this documentation is generated with
[TensorFlow Docs](https://github.com/tensorflow/docs) using the logic defined by
the
[explicit_package_contents_filter](https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/api_generator/public_api.py;l=156)

#### Private TFF API

The TFF API that that is *not* exposed in the TFF
[TFF API documentation](https://www.tensorflow.org/federated/api_docs/python/tff).

#### TFF Python package

The Python [package](https://pypi.org/project/tensorflow-federated/) distributed
on [PyPI](https://pypi.org).

Please be aware, the Python package contains both
[public TFF API](#public-tff-api) and [private TFF API](#private-tff-api) and it
is not obvious *by inspecting the package* which API is intended to be public
and which is intended to be private, for example:

```python
import tensorflow_federated as tff

tff.Computation  # Public TFF API
tff.proto.v0.computation_pb2.Computation  # Private TFF API
```

Therefore, it is useful to keep the
[TFF API documentation](https://www.tensorflow.org/federated/api_docs/python/tff)
in mind when using TFF.

### Diagram

```dot
<!--#include file="package_structure.dot"-->
```

The **green** nodes indicate directories that are part of the
[TFF repository](https://github.com/tensorflow/federated) on
[GitHub](https://github.com) that use the [public TFF API](#public-tff-api).

The **blue** nodes indicate packages that are part of the
[public TFF API](#public-tff-api).

The **grey** nodes indicate directories or packages that are not part of the
[public TFF API](#public-tff-api).

## Details

### Using TFF

#### Research

The `research/` subdirectory has been moved to the
[`federated_research`](https://github.com/google-research/federated) repository,
which contains research projects that use TFF.

#### Examples

The [examples](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/examples)
directory contains examples of how to use TFF.

#### Tests

The [tests](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/tests)
directory contains end-to-end tests of the TFF Python package.

### TFF

[tff](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/) The TensorFlow Federated
library.

#### TFF Simulation

[simulation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/simulation)
Libraries for running TensorFlow Federated simulations.

[simulation/baselines](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/simulation/baselines)
Libraries for constructing TensorFlow Federated baseline tasks.

[simulation/datasets](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/simulation/datasets)
Datasets for running TensorFlow Federated simulations.

[simulation/models](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/simulation/models)
Models for running TensorFlow Federated simulations.

#### TFF Learning

[learning](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning)
Libraries for using Federated Learning algorithms.

[learning/framework](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/framework)
Libraries for developing Federated Learning algorithms.

[learning/optimizers](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/optimizers)
Libraries for optimization algorithms.

[learning/reconstruction](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/reconstruction)
Libraries for using Federated Reconstruction algorithms.

#### TFF Analytics

[analytics](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/analytics)
Libraries for using Federated Analytics algorithms.

#### TFF Aggregators

[aggregators](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/aggregators)
Libraries for constructing federated aggregations.

#### TFF Core

[core](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core) package The
TensorFlow Federated core library.

[core/backends](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/backends)
Backends for constructing, compiling, and executing computations.

[core/templates](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/templates)
Templates for commonly used computations.

[core/test](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/test)
Libraries for testing TensorFlow Federated.

[core/api](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/api)
Libraries for using the [TensorFlow Federated core library](#tff-core).

[core/framework](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/framework)
Libraries for extending the [TensorFlow Federated core library](#tff-core).

[core/types](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/types)
Libraries for interacting with the type of a computation.

#### TFF Impl

[impl](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl) The
implementation of the [TensorFlow Federated core library](#tff-core).

[impl/wrappers](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/wrappers)
Decorators for constructing computations.

[impl/executors](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors)
Libraries for executing computations.

[impl/federated_context](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/federated_context)
Libraries for interacting with a federated context.

[impl/tensorflow_context](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/tensorflow_context)
Libraries for interacting with a TensorFlow context.

[impl/computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/computation)
Libraries for interacting with a computation.

[impl/compiler](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler)
Libraries for compiling computations.

[impl/context_stack](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack)
Libraries for interacting with the context of a computation

[impl/utils](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/utils)
Libraries for use in TensorFlow Federated core library.

[impl/types](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/types)
Libraries for interacting with the type of a computation.

#### TFF Proto

[proto](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/proto) Protobuf
libraries for use in TensorFlow Federated core library.

#### TFF Common Libs

[common_libs](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/common_libs)
Libraries that extend Python for use in TensorFlow Federated.

#### TFF Tensorflow Libs

[tensorflow_libs](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/tensorflow_libs)
Libraries that extend TensorFlow for use in TensorFlow Federated.
