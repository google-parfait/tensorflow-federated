<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.Placement" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="proto"/>
<meta itemprop="property" content="tff_repr"/>
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="uri"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_proto"/>
</div>

# tff.framework.Placement

## Class `Placement`

A representation of a placement literal in TFF's internal language.

Defined in
[`core/impl/computation_building_blocks.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py).

<!-- Placeholder for "Used in" -->

Currently this can only be
<a href="../../tff.md#SERVER"><code>tff.SERVER</code></a> or
<a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(literal)
```

Constructs a new placement instance for the given placement literal.

#### Args:

*   <b>`literal`</b>: The placement literal.

#### Raises:

*   <b>`TypeError`</b>: if the arguments are of the wrong types.

## Properties

<h3 id="proto"><code>proto</code></h3>

Returns a serialized form of this object as a pb.Computation instance.

<h3 id="tff_repr"><code>tff_repr</code></h3>

Returns the representation of the instance using TFF syntax.

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

<h3 id="uri"><code>uri</code></h3>

## Methods

<h3 id="from_proto"><code>from_proto</code></h3>

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
