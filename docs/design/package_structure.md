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
