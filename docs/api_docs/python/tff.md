<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="CLIENTS"/>
<meta itemprop="property" content="SERVER"/>
</div>

# Module: tff

TensorFlow Federated library.

## Modules

[`learning`](./tff/learning.md) module: The public API for model developers using federated learning algorithms.

[`simulation`](./tff/simulation.md) module: The public API for experimenters running federated learning simulations.

[`utils`](./tff/utils.md) module: Utility classes/functions built on top of TensorFlow Federated Core API.

## Classes

[`class Computation`](./tff/Computation.md): An abstract interface for all classes that represent computations.

[`class FederatedType`](./tff/FederatedType.md): An implementation of
<a href="./tff/Type.md"><code>tff.Type</code></a> representing federated types
in TFF.

[`class FunctionType`](./tff/FunctionType.md): An implementation of
<a href="./tff/Type.md"><code>tff.Type</code></a> representing functional types
in TFF.

[`class NamedTupleType`](./tff/NamedTupleType.md): An implementation of
<a href="./tff/Type.md"><code>tff.Type</code></a> representing named tuple types
in TFF.

[`class SequenceType`](./tff/SequenceType.md): An implementation of
<a href="./tff/Type.md"><code>tff.Type</code></a> representing types of
sequences in TFF.

[`class TensorType`](./tff/TensorType.md): An implementation of
<a href="./tff/Type.md"><code>tff.Type</code></a> representing types of tensors
in TFF.

[`class Type`](./tff/Type.md): An abstract interface for all classes that represent TFF types.

[`class Value`](./tff/Value.md): An abstract base class for all values in the bodies of TFF computations.

## Functions

[`federated_aggregate(...)`](./tff/federated_aggregate.md): Aggregates `value` from `CLIENTS` to `SERVER` using a multi-stage process.

[`federated_apply(...)`](./tff/federated_apply.md): Applies a given function to
a federated value on the `SERVER`.

[`federated_average(...)`](./tff/federated_average.md): Computes a `SERVER` average of `value` placed on `CLIENTS`.

[`federated_broadcast(...)`](./tff/federated_broadcast.md): Broadcasts a federated value from the `SERVER` to the `CLIENTS`.

[`federated_collect(...)`](./tff/federated_collect.md): Materializes a federated value from `CLIENTS` as a `SERVER` sequence.

[`federated_computation(...)`](./tff/federated_computation.md): Decorates/wraps Python functions as TFF federated/composite computations.

[`federated_map(...)`](./tff/federated_map.md): Maps a federated value on CLIENTS pointwise using a given mapping function.

[`federated_reduce(...)`](./tff/federated_reduce.md): Reduces `value` from `CLIENTS` to `SERVER` using a reduction operator `op`.

[`federated_sum(...)`](./tff/federated_sum.md): Computes a sum at `SERVER` of a federated value placed on the `CLIENTS`.

[`federated_value(...)`](./tff/federated_value.md): Returns a federated value at
`placement`, with `value` as the constituent.

[`federated_zip(...)`](./tff/federated_zip.md): Converts an N-tuple of federated values into a federated N-tuple value.

[`tf_computation(...)`](./tff/tf_computation.md): Decorates/wraps Python functions and defuns as TFF TensorFlow computations.

[`to_type(...)`](./tff/to_type.md): Converts the argument into an instance of <a href="./tff/Type.md"><code>tff.Type</code></a>.

[`to_value(...)`](./tff/to_value.md): Converts the argument into an instance of the abstract class <a href="./tff/Value.md"><code>tff.Value</code></a>.

## Other Members

<h3 id="CLIENTS"><code>CLIENTS</code></h3>

<h3 id="SERVER"><code>SERVER</code></h3>

