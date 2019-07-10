<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.Selection" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="index"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="proto"/>
<meta itemprop="property" content="source"/>
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_proto"/>
</div>

# tff.framework.Selection

## Class `Selection`

A selection by name or index from a tuple-typed value in TFF's language.

Inherits From:
[`ComputationBuildingBlock`](../../tff/framework/ComputationBuildingBlock.md)

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py>View
source</a>

<!-- Placeholder for "Used in" -->

The concise syntax for selections is `foo.bar` (selecting a named `bar` from the
value of expression `foo`), and `foo[n]` (selecting element at index `n` from
the value of `foo`).

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py>View
source</a>

```python
__init__(
    source,
    name=None,
    index=None
)
```

A selection from 'source' by a string or numeric 'name_or_index'.

Exactly one of 'name' or 'index' must be specified (not None).

#### Args:

*   <b>`source`</b>: The source value to select from (an instance of
    ComputationBuildingBlock).
*   <b>`name`</b>: A string name of the element to be selected.
*   <b>`index`</b>: A numeric index of the element to be selected.

#### Raises:

*   <b>`TypeError`</b>: if arguments are of the wrong types.
*   <b>`ValueError`</b>: if the name is empty or index is negative, or the
    name/index is not compatible with the type signature of the source, or
    neither or both are defined (not None).

## Properties

<h3 id="index"><code>index</code></h3>

<h3 id="name"><code>name</code></h3>

<h3 id="proto"><code>proto</code></h3>

Returns a serialized form of this object as a pb.Computation instance.

<h3 id="source"><code>source</code></h3>

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

## Methods

<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py>View
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
