# Backend

[TOC]

A backend is the composition of a [compiler](compilation.md#compiler) and a
[runtime](execution.md#runtime) in a [Context](context.md#context) used to
[construct](tracing.md), [compile](compilation.md), and [execute](execution.md)
an [AST](compilation.md#ast), meaning a backend constructs environments that
evaluate an AST.

The
[backends](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends)
package contains backends which may extend the TFF compiler and/or the TFF
runtime; these extensions can be found in the corresponding backend.

If the [runtime](execution.md#runtime) of a backend is implemented as an
[execution stack](execution.md#execution-stack), then the backend can construct
an [ExecutionContext](context.md#executioncontext) to provide TFF with an
environemnt in which to evaluate an AST. In this case, the backend is
integrating with TFF using the high-level abstraction. However, if the runtime
is *not* implemented as an execution stack, then the backend will need to
construct a [Context](context.md#context) and is integrating with TFF using the
low-level abstraction.

```dot
<!--#include file="backend.dot"-->
```

The **blue** nodes are provided by TFF
[core](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core).

The **green**, **red**, **yellow**, and **purple** nodes are provided by the
[native](#native), [mapreduce](#mapreduce), and [reference](#reference) backends
respectively.

The **dashed** nodes are provided by an external system.

The **solid** arrows indicate relationship and the **dashed** arrows indicate
inheritance.

## Native

The
[native](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native)
backend composes of the TFF compiler and TFF runtime in order to compile and
execute an AST in a way that is reasonably efficiant and debuggable.

### Native Form

A native form is an AST that is topologically sorted into a directed acyclic
graph (DAG) of TFF intrinsics with some optimizations to the dependency of those
intrinsics.

### Compiler

The
[compiler.transform_to_native_form](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/compiler.py)
function compiles an AST into a [native form](#native-form).

### Runtime

The native backend does not contain backend specific extentions to the TFF
runtime, instead an [execution stack](execution.md#execution-stack) can be used
directly.

### Context

A native context is an [ExecutionContext](context.md#executioncontext)
constructed with a native compiler (or no compiler) and a TFF runtime, for
example:

```python
executor = eager_tf_executor.EagerTFExecutor()
factory = executor_factory.create_executor_factory(lambda _: executor)
context = execution_context.ExecutionContext(
    executor_fn=factory,
    compiler_fn=None)
set_default_context.set_default_context(context)
```

However, there are some common configurations:

The
[execution_context.set_sync_local_cpp_execution_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/execution_context.py)
function constructs an `ExecutionContext` with a native compiler and a
[local execution stack](execution.md#local-execution-stack).

## MapReduce

The
[mapreduce](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce)
backend contains the data structures and compiler required to construct a form
that can be executed on MapReduce-like runtimes.

### `MapReduceForm`

A
[forms.MapReduceForm](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py)
is a data structure defining the representation of logic that can be executed on
MapReduce-like runtimes. This logic is organized as a collection of TensorFlow
functions, see the
[forms](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py)
module for more information about the nature of these functions.

### Compiler

The
[compiler](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/compiler.py)
module contains [Building Block](compilation.md#building-block) and
[TensorFlow Computation](compilation.md#tensorflow-computation) transformations
required to compile an AST to a [MapReduceForm](#canonicalform).

The
[form_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/form_utils.py)
module contains the compiler for the MapReduce backend and constructs a
[MapReduceForm](#canonicalform).

### Runtime

A MapReduce runtime is not provided by TFF, instead this should be provided by
an external MapReduce-like system.

### Context

A MapReduce context is not provided by TFF.
