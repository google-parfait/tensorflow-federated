<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.ComputationBuildingBlock" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="proto"/>
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="compact_representation"/>
<meta itemprop="property" content="formatted_representation"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="structural_representation"/>
</div>

# tff.framework.ComputationBuildingBlock

## Class `ComputationBuildingBlock`

The abstract base class for abstractions in the TFF's internal language.

Inherits From: [`TypedObject`](../../tff/TypedObject.md)

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
source</a>

<!-- Placeholder for "Used in" -->

Instances of this class correspond roughly one-to-one to the abstractions
defined in the `Computation` message in TFF's `computation.proto`, and are
intended primarily for the ease of manipulating the abstract syntax trees (AST)
of federated computations as they are transformed by TFF's compiler pipeline to
mold into the needs of a particular execution backend. The only abstraction that
does not have a dedicated Python equivalent is a section of TensorFlow code
(it's represented by
<a href="../../tff/framework/CompiledComputation.md"><code>tff.framework.CompiledComputation</code></a>).

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
source</a>

```python
__init__(type_spec)
```

Constructs a computation building block with the given TFF type.

#### Args:

*   <b>`type_spec`</b>: An instance of types.Type, or something convertible to
    it via types.to_type().

## Properties

<h3 id="proto"><code>proto</code></h3>

Returns a serialized form of this object as a pb.Computation instance.

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

## Methods

<h3 id="compact_representation"><code>compact_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
source</a>

```python
compact_representation()
```

Returns the compact string representation of this building block.

<h3 id="formatted_representation"><code>formatted_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
source</a>

```python
formatted_representation()
```

Returns the formatted string representation of this building block.

<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
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

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
source</a>

```python
structural_representation()
```

Returns the structural string representation of this building block.
