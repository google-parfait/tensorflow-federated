# Compilation

[TOC]

The
[compiler](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler)
package contains data structures defining the Python representation of the
[AST](#ast), core [transformation](#transformation) functions, and
[compiler](#compiler) related functionality.

## AST

An abstract syntax tree (AST) in TFF describes the structure of a federated
computation.

### Building Block

A
[building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
is the Python representation of the [AST](#ast).

#### `CompiledComputation`

A
[building_block.CompiledComputation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
is a
[building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
that represents a computation that will be delegated to an
[external runtime](execution.md#external-runtime). Currently TFF only supports
[TensorFlow computations](#tensorFlow-computation), but could be expanded to
support [Computations](#computation) backed by other external runtimes.

### `Computation`

A
[pb.Computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto)
is the Proto or serialized representation of the [AST](#ast).

#### TensorFlow Computation

A
[pb.Computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto)
that represents a [Computations](#computation) that will be delegated to the
[TensorFlow](execution.md#tensorflow) runtime.

## Transformation

A transformation constructs a new [AST](#ast) for a given AST after applying a
collection of mutations. Transformations can operate on
[building blocks](#building-block) in order to transform the Python
representation of the AST or on
[TensorFlow computations](#tensorFlow-computation) in order to transform a
`tf.Graph`.

An **atomic** transformation is one that applies a single mutation (possibly
more than once) to the given input.

A **composite** transformation is one that applies multiple transformations to
the given input in order to provide some feature or assertion.

Note: Transformations can be composed in serial or parallel, meaning that you
can construct a composite transformation that performs multiple transformations
in one pass through an AST. However, the order in which you apply
transformations and how those transformations are parallelized is hard to reason
about; as a result, composite transformations are hand-crafted and most are
somewhat fragile.

The
[tree_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tree_transformations.py)
module contains atomic [building block](#building-block) transformations.

The
[transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformations.py)
module contains composite [building block](#building-block) transformations.

The
[tensorflow_computation_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tensorflow_computation_transformations.py)
module contains atomic [TensorFlow computation](#tensorflow-computation)
transformations.

The
[compiled_computation_transforms](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/compiled_computation_transforms.py)
module contains atomic and composite
[Compiled Computation](#compiled-computation) transformations.

The
[transformation_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformation_utils.py)
module contains functions, traversal logic, and data structures used by other
transformation modules.

## Compiler

A compiler is a collection of [transformations](#transformation) that construct
a form that can be executed.
