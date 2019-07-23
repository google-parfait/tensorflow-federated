<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.Tuple" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="proto"/>
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="compact_representation"/>
<meta itemprop="property" content="formatted_representation"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="structural_representation"/>
</div>

# tff.framework.Tuple

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
source</a>

## Class `Tuple`

A tuple with named or unnamed elements in TFF's internal language.

Inherits From:
[`ComputationBuildingBlock`](../../tff/framework/ComputationBuildingBlock.md)

<!-- Placeholder for "Used in" -->

The concise notation for tuples is `<name_1=value_1, ...., name_n=value_n>` for
tuples with named elements, `<value_1, ..., value_n>` for tuples with unnamed
elements, or a mixture of these for tuples with ome named and some unnamed
elements, where `name_k` are the names, and `value_k` are the value expressions.

For example, a lambda expression that applies `fn` to elements of 2-tuples
pointwise could be represented as `(arg -> <fn(arg[0]),fn(arg[1])>)`.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py">View
source</a>

```python
__init__(elements)
```

Constructs a tuple from the given list of elements.

#### Args:

*   <b>`elements`</b>: The elements of the tuple, supplied as a list of (name,
    value) pairs, where 'name' can be None in case the corresponding element is
    not named and only accessible via an index (see also AnonymousTuple).

#### Raises:

*   <b>`TypeError`</b>: if arguments are of the wrong types.

## Properties

<h3 id="proto"><code>proto</code></h3>

Returns a serialized form of this object as a pb.Computation instance.

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/common_libs/anonymous_tuple.py">View
source</a>

```python
__eq__(other)
```

Return self==value.

<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/common_libs/anonymous_tuple.py">View
source</a>

```python
__getitem__(key)
```

<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/common_libs/anonymous_tuple.py">View
source</a>

```python
__iter__()
```

<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/common_libs/anonymous_tuple.py">View
source</a>

```python
__len__()
```

<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/common_libs/anonymous_tuple.py">View
source</a>

```python
__ne__(other)
```

Return self!=value.

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
