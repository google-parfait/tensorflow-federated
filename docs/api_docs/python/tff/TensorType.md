<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.TensorType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="shape"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tff.TensorType

## Class `TensorType`

Inherits From: [`Type`](../tff/Type.md)

Defined in
[`core/api/computation_types.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py).

An implementation of <a href="../tff/Type.md"><code>tff.Type</code></a>
representing types of tensors in TFF.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    dtype,
    shape=None
)
```

Constructs a new instance from the given `dtype` and `shape`.

#### Args:

*   <b>`dtype`</b>: An instance of `tf.DType`.
*   <b>`shape`</b>: An optional instance of `tf.TensorShape` or an argument that
    can be passed to its constructor (such as a `list` or a `tuple`), or `None`
    for the default scalar shape. Unspecified shapes are not supported.

#### Raises:

*   <b>`TypeError`</b>: if arguments are of the wrong types.

## Properties

<h3 id="dtype"><code>dtype</code></h3>

<h3 id="shape"><code>shape</code></h3>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

```python
__eq__(other)
```

<h3 id="__ne__"><code>__ne__</code></h3>

```python
__ne__(other)
```
