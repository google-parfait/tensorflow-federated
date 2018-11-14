<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff

TensorFlow Federated library.

## Classes

[`class Computation`](./tff/Computation.md): An abstract interface for all classes that represent computations.

[`class FunctionType`](./tff/FunctionType.md): An implementation of Type for representing functional types in TFF.

[`class NamedTupleType`](./tff/NamedTupleType.md): An implementation of Type for representing types of named tuples in TFF.

[`class SequenceType`](./tff/SequenceType.md): An implementation of Type for representing types of sequences in TFF.

[`class TensorType`](./tff/TensorType.md): An implementation of Type for representing types of tensors in TFF.

[`class Type`](./tff/Type.md): An abstract interface for all classes that represent TFF types.

## Functions

[`federated_computation(...)`](./tff/federated_computation.md): Decorates/wraps Python functions as TFF federated/composite computations.

[`tf_computation(...)`](./tff/tf_computation.md): Decorates/wraps Python functions and defuns as TFF TensorFlow computations.

[`to_type(...)`](./tff/to_type.md): Converts the argument into an instance of Type.

