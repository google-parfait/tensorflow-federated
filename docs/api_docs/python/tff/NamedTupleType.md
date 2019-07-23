<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.NamedTupleType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="compact_representation"/>
<meta itemprop="property" content="formatted_representation"/>
</div>

# tff.NamedTupleType

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

## Class `NamedTupleType`

An implementation of <a href="../tff/Type.md"><code>tff.Type</code></a>
representing named tuple types in TFF.

Inherits From: [`Type`](../tff/Type.md)

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
__init__(elements)
```

Constructs a new instance from the given element types.

#### Args:

*   <b>`elements`</b>: Element specifications, in the format of a `list`,
    `tuple`, or `collections.OrderedDict`. Each element specification is either
    a type spec (an instance of
    <a href="../tff/Type.md"><code>tff.Type</code></a> or something convertible
    to it via <a href="../tff/to_type.md"><code>tff.to_type</code></a>) for the
    element, or a (name, spec) for elements that have defined names.
    Alternatively, one can supply here an instance of `collections.OrderedDict`
    mapping element names to their types (or things that are convertible to
    types).

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
__eq__(other)
```

Determines whether two type definitions are identical.

Note that this notion of equality is stronger than equivalence. Two types with
equivalent definitions may not be identical, e.g., if they represent templates
with differently named type variables in their definitions.

#### Args:

*   <b>`other`</b>: The other type to compare against.

#### Returns:

`True` iff type definitions are syntatically identical (as defined above), or
`False` otherwise.

#### Raises:

*   <b>`NotImplementedError`</b>: If not implemented in the derived class.

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

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
compact_representation()
```

Returns the compact string representation of this type.

<h3 id="formatted_representation"><code>formatted_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
formatted_representation()
```

Returns the formatted string representation of this type.
