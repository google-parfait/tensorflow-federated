<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.Type" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tff.Type

## Class `Type`

Defined in
[`core/api/computation_types.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py).

An abstract interface for all classes that represent TFF types.

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```

Determines whether two type definitions are identical.

Note that this notion of equality is stronger than equivalence. Two types
with equivalent definitions may not be identical, e.g., if they represent
templates with differently named type veriables in their definitions.

#### Args:

* <b>`other`</b>: The other type to compare against.


#### Returns:

`True` iff type definitions are syntatically identical (as defined above),
or `False` otherwise.


#### Raises:

* <b>`NotImplementedError`</b>: If not implemented in the derived class.

<h3 id="__ne__"><code>__ne__</code></h3>

``` python
__ne__(other)
```
