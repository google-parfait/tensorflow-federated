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

Inherits From: [`Type`](../tff/Type.md)

Defined in
[`core/api/computation_types.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py).

An implementation of <a href="../tff/Type.md"><code>tff.Type</code></a>
representing types of sequences in TFF.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(element)
```

Constructs a new instance from the given element type.

#### Args:

*   <b>`element`</b>: A specification of the element type, either an instance of
    <a href="../tff/Type.md"><code>tff.Type</code></a> or something convertible
    to it by `tff.to_type()`.

## Properties

<h3 id="element"><code>element</code></h3>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

```python
__eq__(other)
```

<h3 id="__ne__"><code>__ne__</code></h3>

```python
__ne__(other)
```
