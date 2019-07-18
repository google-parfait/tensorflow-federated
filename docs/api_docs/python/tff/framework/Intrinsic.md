<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.Intrinsic" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="proto"/>
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="uri"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_proto"/>
</div>

# tff.framework.Intrinsic

## Class `Intrinsic`

A representation of an intrinsic in TFF's internal language.

Inherits From:
[`ComputationBuildingBlock`](../../tff/framework/ComputationBuildingBlock.md)

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
source</a>

<!-- Placeholder for "Used in" -->

An instrinsic is a symbol known to the TFF's compiler pipeline, represended a a
known URI. It generally appears in expressions with a concrete type, although
all intrinsic are defined with template types. This class does not deal with
parsing intrinsic URIs and verifying their types, it is only a container.
Parsing and type analysis are a responsibility or the components that manipulate
ASTs.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
source</a>

```python
__init__(
    uri,
    type_spec
)
```

Creates an intrinsic.

#### Args:

*   <b>`uri`</b>: The URI of the intrinsic.
*   <b>`type_spec`</b>: Either the types.Type that represents the type of this
    intrinsic, or something convertible to it by types.to_type().

#### Raises:

*   <b>`TypeError`</b>: if the arguments are of the wrong types.

## Properties

<h3 id="proto"><code>proto</code></h3>

Returns a serialized form of this object as a pb.Computation instance.

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

<h3 id="uri"><code>uri</code></h3>

## Methods

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
