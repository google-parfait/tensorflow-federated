<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="CLIENTS"/>
<meta itemprop="property" content="SERVER"/>
<meta itemprop="property" content="federated_broadcast"/>
<meta itemprop="property" content="federated_map"/>
<meta itemprop="property" content="federated_sum"/>
</div>

# Module: tff

TensorFlow Federated library.

## Classes

[`class Computation`](./tff/Computation.md): An abstract interface for all classes that represent computations.

[`class FederatedType`](./tff/FederatedType.md): An implementation of `Type` for representing federated types in TFF.

[`class FunctionType`](./tff/FunctionType.md): An implementation of `Type` for representing functional types in TFF.

[`class NamedTupleType`](./tff/NamedTupleType.md): An implementation of `Type` for representing named tuple types in TFF.

[`class SequenceType`](./tff/SequenceType.md): An implementation of `Type` for representing types of sequences in TFF.

[`class TensorType`](./tff/TensorType.md): An implementation of `Type` for representing types of tensors in TFF.

[`class Type`](./tff/Type.md): An abstract interface for all classes that represent TFF types.

## Functions

[`federated_computation(...)`](./tff/federated_computation.md): Decorates/wraps Python functions as TFF federated/composite computations.

[`tf_computation(...)`](./tff/tf_computation.md): Decorates/wraps Python functions and defuns as TFF TensorFlow computations.

[`to_type(...)`](./tff/to_type.md): Converts the argument into an instance of `Type`.

## Other Members

<h3 id="CLIENTS"><code>CLIENTS</code></h3>

<h3 id="SERVER"><code>SERVER</code></h3>

<h3 id="federated_broadcast"><code>federated_broadcast</code></h3>

<h3 id="federated_map"><code>federated_map</code></h3>

<h3 id="federated_sum"><code>federated_sum</code></h3>

