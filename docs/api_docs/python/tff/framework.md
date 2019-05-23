<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.framework

Interfaces for extensions, selectively lifted out of `impl`.

Defined in
[`core/framework/__init__.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/framework/__init__.py).

<!-- Placeholder for "Used in" -->

## Classes

[`class Block`](../tff/framework/Block.md): A representation of a block of code
in TFF's internal language.

[`class Call`](../tff/framework/Call.md): A representation of a function
invocation in TFF's internal language.

[`class CompiledComputation`](../tff/framework/CompiledComputation.md): A
representation of a fully constructed and serialized computation.

[`class ComputationBuildingBlock`](../tff/framework/ComputationBuildingBlock.md):
The abstract base class for abstractions in the TFF's internal language.

[`class Intrinsic`](../tff/framework/Intrinsic.md): A representation of an
intrinsic in TFF's internal language.

[`class Lambda`](../tff/framework/Lambda.md): A representation of a lambda
expression in TFF's internal language.

[`class Placement`](../tff/framework/Placement.md): A representation of a
placement literal in TFF's internal language.

[`class Reference`](../tff/framework/Reference.md): A reference to a name
defined earlier in TFF's internal language.

[`class Selection`](../tff/framework/Selection.md): A selection by name or index
from a tuple-typed value in TFF's language.

[`class Tuple`](../tff/framework/Tuple.md): A tuple with named or unnamed
elements in TFF's internal language.

## Functions

[`is_assignable_from(...)`](../tff/framework/is_assignable_from.md): Determines
whether `target_type` is assignable from `source_type`.
