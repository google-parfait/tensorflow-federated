# Execution

[TOC]

The
[executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors)
package contains core [Executors](#executor) classes and [runtime](#runtime)
related functionality.

## Runtime

A runtime is a logical concept describing a system that executes a computation.

### TFF Runtime

A TFF runtime typically handles executing an [AST](compilation.md#ast) and
delegates executing mathematical computations to a
[external runtime](#external-runtime) such as [TensorFlow](#tensorflow).

### External Runtime

An external runtime is any system that the TFF runtime delegates execution to.

#### TensorFlow

[TensorFlow](https://www.tensorflow.org/) is an open source platform for machine
learning. Today the TFF runtime delegates mathematical computations to
TensorFlow using an [Executor](#Executor) that can be composed into a hierarchy,
referred to as an [execution stack](#execution-stack).

## `Executor`

An
[executor_base.Executor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_base.py)
is an abstract interface that defines the API for executing an
[AST](compilation.md#ast). The
[executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors)
package contains a collection of concrete implementations of this interface.

## `ExecutorFactory`

An
[executor_factory.ExecutorFactory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_factory.py)
is an abstract interface that defines the API for constructing an
[Executor](#executor). These factories construct the executor lazily and manage
the lifecycle of the executor; the motivation to lazily constructing executors
is to infer the number of clients at execution time.

## Execution Stack

An execution stack is a hierarchy of [Executors](#executor). The
[executor_stacks](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executor_stacks)
package contains logic for constructing and composing specific execution stacks.
