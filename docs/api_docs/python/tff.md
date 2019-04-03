<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="CLIENTS"/>
<meta itemprop="property" content="SERVER"/>
</div>

# Module: tff

Defined in
[`__init__.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/__init__.py).

<!-- Placeholder for "Used in" -->

TensorFlow Federated library.

## Modules

[`learning`](./tff/learning.md) module: The public API for model developers
using federated learning algorithms.

[`simulation`](./tff/simulation.md) module: The public API for experimenters
running federated learning simulations.

[`utils`](./tff/utils.md) module: Utility classes/functions built on top of
TensorFlow Federated Core API.

## Classes

[`class Computation`](./tff/Computation.md): An abstract interface for all
classes that represent computations.

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

[`class Type`](./tff/Type.md): An abstract interface for all classes that
represent TFF types.

[`class TypedObject`](./tff/TypedObject.md): An abstract interface for things
that possess TFF type signatures.

[`class Value`](./tff/Value.md): An abstract base class for all values in the
bodies of TFF computations.

## Functions

[`federated_aggregate(...)`](./tff/federated_aggregate.md): Aggregates `value`
from <a href="./tff.md#CLIENTS"><code>tff.CLIENTS</code></a> to
<a href="./tff.md#SERVER"><code>tff.SERVER</code></a>.

[`federated_apply(...)`](./tff/federated_apply.md): Applies a given function to
a federated value on the <a href="./tff.md#SERVER"><code>tff.SERVER</code></a>.

[`federated_broadcast(...)`](./tff/federated_broadcast.md): Broadcasts a
federated value from the <a href="./tff.md#SERVER"><code>tff.SERVER</code></a>
to the <a href="./tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

[`federated_collect(...)`](./tff/federated_collect.md): Returns a federated
value from <a href="./tff.md#CLIENTS"><code>tff.CLIENTS</code></a> as a
<a href="./tff.md#SERVER"><code>tff.SERVER</code></a> sequence.

[`federated_computation(...)`](./tff/federated_computation.md): Decorates/wraps
Python functions as TFF federated/composite computations.

[`federated_map(...)`](./tff/federated_map.md): Maps a federated value on
<a href="./tff.md#CLIENTS"><code>tff.CLIENTS</code></a> pointwise using a
mapping function.

[`federated_mean(...)`](./tff/federated_mean.md): Computes a
<a href="./tff.md#SERVER"><code>tff.SERVER</code></a> mean of `value` placed on
<a href="./tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

[`federated_reduce(...)`](./tff/federated_reduce.md): Reduces `value` from
<a href="./tff.md#CLIENTS"><code>tff.CLIENTS</code></a> to
<a href="./tff.md#SERVER"><code>tff.SERVER</code></a> using a reduction `op`.

[`federated_sum(...)`](./tff/federated_sum.md): Computes a sum at
<a href="./tff.md#SERVER"><code>tff.SERVER</code></a> of a `value` placed on the
<a href="./tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

[`federated_value(...)`](./tff/federated_value.md): Returns a federated value at
`placement`, with `value` as the constituent.

[`federated_zip(...)`](./tff/federated_zip.md): Converts an N-tuple of federated
values into a federated N-tuple value.

[`sequence_map(...)`](./tff/sequence_map.md): Maps a TFF sequence `value`
pointwise using a given function `mapping_fn`.

[`sequence_reduce(...)`](./tff/sequence_reduce.md): Reduces a TFF sequence
`value` given a `zero` and reduction operator `op`.

[`sequence_sum(...)`](./tff/sequence_sum.md): Computes a sum of elements in a
sequence.

[`tf_computation(...)`](./tff/tf_computation.md): Decorates/wraps Python
functions and defuns as TFF TensorFlow computations.

[`to_type(...)`](./tff/to_type.md): Converts the argument into an instance of
<a href="./tff/Type.md"><code>tff.Type</code></a>.

[`to_value(...)`](./tff/to_value.md): Converts the argument into an instance of
the abstract class <a href="./tff/Value.md"><code>tff.Value</code></a>.

## Other Members

<h3 id="CLIENTS"><code>CLIENTS</code></h3>

<h3 id="SERVER"><code>SERVER</code></h3>
