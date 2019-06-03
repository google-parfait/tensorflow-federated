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

An implementation of <a href="../tff/Type.md"><code>tff.Type</code></a>
representing functional types in TFF.

Inherits From: [`Type`](../tff/Type.md)

Defined in
[`python/core/api/computation_types.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py).

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    parameter,
    result
)
```

Constructs a new instance from the given `parameter` and `result` types.

#### Args:

*   <b>`parameter`</b>: A specification of the parameter type, either an
    instance of <a href="../tff/Type.md"><code>tff.Type</code></a> or something
    convertible to it by
    <a href="../tff/to_type.md"><code>tff.to_type</code></a>.
*   <b>`result`</b>: A specification of the result type, either an instance of
    <a href="../tff/Type.md"><code>tff.Type</code></a> or something convertible
    to it by <a href="../tff/to_type.md"><code>tff.to_type</code></a>.

## Properties

<h3 id="parameter"><code>parameter</code></h3>

<h3 id="result"><code>result</code></h3>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

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

```python
__ne__(other)
```

Return self!=value.
