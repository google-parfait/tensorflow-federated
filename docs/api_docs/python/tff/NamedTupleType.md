<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.NamedTupleType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__dir__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getattr__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tff.NamedTupleType

## Class `NamedTupleType`

Inherits From: [`Type`](../tff/Type.md)

An implementation of `Type` for representing named tuple types in TFF.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(elements)
```

Constructs a new instance from the given element types.

#### Args:

* <b>`elements`</b>: Element specifications, in the format of a list, OrderedDict, or
    tuple. Each element specification is either a type spec (an instance of
    `Type` or something convertible to it via `to_type()`) for the element,
    or a (name, spec) for elements that have defined names. Alternatively
    one can supply here an instance of `collections.OrderedDict` mapping
    element names to their types (or things that are convertible to types).


#### Raises:

* <b>`TypeError`</b>: if the arguments are of the wrong types.
* <b>`ValueError`</b>: if the named tuple contains no elements.



## Methods

<h3 id="__dir__"><code>__dir__</code></h3>

``` python
__dir__()
```



<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```



<h3 id="__getattr__"><code>__getattr__</code></h3>

``` python
__getattr__(name)
```



<h3 id="__getitem__"><code>__getitem__</code></h3>

``` python
__getitem__(key)
```



<h3 id="__iter__"><code>__iter__</code></h3>

``` python
__iter__()
```



<h3 id="__len__"><code>__len__</code></h3>

``` python
__len__()
```



<h3 id="__ne__"><code>__ne__</code></h3>

``` python
__ne__(other)
```





