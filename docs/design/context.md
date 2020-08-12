# Context

[TOC]

## `Context`

A
[context_base.Context](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/context_base.py)
is an environment that can [construct](tracing.md), [compile](compilation.md),
or [execute](execution.md) an [AST](compilation.md#ast).

This API defines a **low-level abstraction** that should be used when an
[Executor](execution.md#executor) is **not** used for execution; the
[Reference](backend.md#reference) backend integrate at this level.

### `ExecutionContext`

An
[execution_context.ExecutionContext](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors/execution_context.py)
is a
[context_base.Context](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/context_base.py)
that compiles computations using a compilation function and executes
computations using an [Executor](execution.md#executor).

This API defines a **high-level abstraction** that should be used when an
[Executor](execution.md#executor) is used for execution; the
[native](backend.md#native) and [IREE](backend.md#iree) backends integrate at
this level.

### `FederatedComputationContext`

A
[federated_computation_context.FederatedComputationContext](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/federated_context/federated_computation_context.py)
is a
[context_base.Context](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/context_base.py)
that constructs federated computations. This context is used trace Python
functions decorated with the
[computations.federated_computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/api/computations.py)
decorator.

### `TensorFlowComputationContext`

A
[tensorflow_computation_context.TensorFlowComputationContext](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation_context.py)
is a
[context_base.Context](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/context_base.py)
that constructs TensorFlow computations. This context is used to serialize
Python functions decorated with the
[computations.tf_computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/api/computations.py)
decorator.

## `ContextStack`

A
[context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py)
is a data structure for interacting with a stack of [Contexts](#context).

You can set the context TFF will use to [construct](tracing.md),
[compile](compilation.md), or [execute](execution.md) an
[AST](compilation.md#ast) by:

*   Invoking
    [set_default_context.set_default_context](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/set_default_context.py)
    to set the default context. This API is often used to install a context that
    will compile or execute a computaiton.

*   Invoking
    [get_context_stack.get_context_stack](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/get_context_stack.py)
    to get the current context stack and then invoking
    [context_stack_base.ContextStack.install](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py)
    to temporarily install a context onto the top of the stack. For example, the
    [computations.federated_computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/api/computations.py)
    and
    [computations.tf_computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/api/computations.py)
    decorators push the corresponding contexts onto the current context stack
    while the decorated function is being traced.

### `ContextStackImpl`

A
[context_stack_impl.ContextStackImpl](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/context_stack_impl.py)
is a
[context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py)
that is implemented as a common thread-local stack.
