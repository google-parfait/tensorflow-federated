<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.NamedTupleType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="elements"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="is_assignable_from"/>
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

* <b>`elements`</b>: A list of element specifications. Each element specification is
    either a type spec (an instance of `Type` or something convertible to it
    via `to_type()`) for the element, or a pair (name, spec) for elements
    that have defined names. Alternatively, one can supply here an instance
    of `collections.OrderedDict` mapping element names to their types (or
    things that are convertible to types).


#### Raises:

* <b>`TypeError`</b>: if the arguments are of the wrong types.
* <b>`ValueError`</b>: if the named tuple contains no elements.



## Properties

<h3 id="elements"><code>elements</code></h3>





## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```

Return self==value.

<h3 id="__ge__"><code>__ge__</code></h3>

``` python
__ge__(other)
```

Return self>=value.

<h3 id="__gt__"><code>__gt__</code></h3>

``` python
__gt__(other)
```

Return self>value.

<h3 id="__le__"><code>__le__</code></h3>

``` python
__le__(other)
```

Return self<=value.

<h3 id="__lt__"><code>__lt__</code></h3>

``` python
__lt__(other)
```

Return self<value.

<h3 id="__ne__"><code>__ne__</code></h3>

``` python
__ne__(other)
```

Return self!=value.

<h3 id="is_assignable_from"><code>is_assignable_from</code></h3>

``` python
is_assignable_from(other)
```

Determines whether this TFF type is assignable from another TFF type.

#### Args:

* <b>`other`</b>: Another type, an instance of `Type`.


#### Returns:

`True` if self is assignable from other, `False` otherwise.



