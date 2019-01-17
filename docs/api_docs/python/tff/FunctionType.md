<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.FunctionType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="parameter"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tff.FunctionType

## Class `FunctionType`

Inherits From: [`Type`](../tff/Type.md)

Defined in
[`core/api/computation_types.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py).

An implementation of <a href="../tff/Type.md"><code>tff.Type</code></a>
representing functional types in TFF.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    parameter,
    result
)
```

Constructs a new instance from the given parameter and result types.

#### Args:

*   <b>`parameter`</b>: A specification of the parameter type, either an
    instance of <a href="../tff/Type.md"><code>tff.Type</code></a> or something
    convertible to it by `tff.to_type()`.
*   <b>`result`</b>: A specification of the result type, either an instance of
    <a href="../tff/Type.md"><code>tff.Type</code></a> or something convertible
    to it by `tff.to_type()`.

## Properties

<h3 id="parameter"><code>parameter</code></h3>

<h3 id="result"><code>result</code></h3>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

```python
__eq__(other)
```

<h3 id="__ne__"><code>__ne__</code></h3>

```python
__ne__(other)
```
