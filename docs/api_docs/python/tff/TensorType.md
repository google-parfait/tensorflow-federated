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

An implementation of `Type` for representing types of tensors in TFF.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    dtype,
    shape=None
)
```

Constructs a new instance from the given dtype and shape.

#### Args:

* <b>`dtype`</b>: An instance of `tf.DType`.
* <b>`shape`</b>: An optional instance of `tf.TensorShape` or an argument that can be
    passed to its constructor (such as a `list` or a `tuple`), or `None` for
    the default scalar shape. Unspecified shapes are not supported.


#### Raises:

* <b>`TypeError`</b>: if arguments are of the wrong types.



## Properties

<h3 id="dtype"><code>dtype</code></h3>



<h3 id="shape"><code>shape</code></h3>





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

Return self!=value.



