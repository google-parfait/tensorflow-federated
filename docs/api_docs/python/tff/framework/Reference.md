<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.Reference" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="context"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="proto"/>
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="compact_representation"/>
<meta itemprop="property" content="formatted_representation"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="structural_representation"/>
</div>

# tff.framework.Reference

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

## Class `Reference`

A reference to a name defined earlier in TFF's internal language.

Inherits From:
[`ComputationBuildingBlock`](../../tff/framework/ComputationBuildingBlock.md)

<!-- Placeholder for "Used in" -->

Names are defined by lambda expressions (which have formal named parameters),
and block structures (which can have one or more locals). The reference
construct is used to refer to those parameters or locals by a string name. The
usual hiding rules apply. A reference binds to the closest definition of the
given name in the most deeply nested surrounding lambda or block.

A concise notation for a reference to name `foo` is `foo`. For example, in a
lambda expression `(x -> f(x))` there are two references, one to `x` that is
defined as the formal parameter of the lambda epxression, and one to `f` that
must have been defined somewhere in the surrounding context.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

```python
__init__(
    name,
    type_spec,
    context=None
)
```

Creates a reference to 'name' of type 'type_spec' in context 'context'.

#### Args:

*   <b>`name`</b>: The name of the referenced entity.
*   <b>`type_spec`</b>: The type spec of the referenced entity.
*   <b>`context`</b>: The optional context in which the referenced entity is
    defined. This class does not prescribe what Python type the 'context' needs
    to be and merely exposes it as a property (see below). The only requirement
    is that the context implements str() and repr().

#### Raises:

*   <b>`TypeError`</b>: if the arguments are of the wrong types.

## Properties

<h3 id="context"><code>context</code></h3>

<h3 id="name"><code>name</code></h3>

<h3 id="proto"><code>proto</code></h3>

Returns a serialized form of this object as a pb.Computation instance.

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

## Methods

<h3 id="compact_representation"><code>compact_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

```python
compact_representation()
```

Returns the compact string representation of this building block.

<h3 id="formatted_representation"><code>formatted_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

```python
formatted_representation()
```

Returns the formatted string representation of this building block.

<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

```python
@classmethod
from_proto(
    cls,
    computation_proto
)
```

Returns an instance of a derived class based on 'computation_proto'.

#### Args:

*   <b>`computation_proto`</b>: An instance of pb.Computation.

#### Returns:

An instance of a class that implements 'ComputationBuildingBlock' and that
contains the deserialized logic from in 'computation_proto'.

#### Raises:

*   <b>`NotImplementedError`</b>: if computation_proto contains a kind of
    computation for which deserialization has not been implemented yet.
*   <b>`ValueError`</b>: if deserialization failed due to the argument being
    invalid.

<h3 id="structural_representation"><code>structural_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

```python
structural_representation()
```

Returns the structural string representation of this building block.
