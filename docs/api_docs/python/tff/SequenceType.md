<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.SequenceType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="element"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tff.SequenceType

## Class `SequenceType`

An implementation of <a href="../tff/Type.md"><code>tff.Type</code></a>
representing types of sequences in TFF.

Inherits From: [`Type`](../tff/Type.md)

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
__init__(element)
```

Constructs a new instance from the given `element` type.

#### Args:

*   <b>`element`</b>: A specification of the element type, either an instance of
    <a href="../tff/Type.md"><code>tff.Type</code></a> or something convertible
    to it by <a href="../tff/to_type.md"><code>tff.to_type</code></a>.

## Properties

<h3 id="element"><code>element</code></h3>

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

<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
__ne__(other)
```

Return self!=value.
