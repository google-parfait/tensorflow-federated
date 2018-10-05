<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.NamedTupleType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="elements"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="is_assignable_from"/>
</div>

# tff.NamedTupleType

## Class `NamedTupleType`

Inherits From: [`Type`](../tff/Type.md)

An implementation of Type for representing types of named tuples in TFF.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(elements)
```

Constructs a new instance from the given element types.

#### Args:

* <b>`elements`</b>: A list of element specifications. Each element specification
    is either a type spec (an instance of Type or something convertible to
    it via to_type() below) for the element, or a pair (name, spec) for
    elements that have defined names. Alternatively, one can supply here
    an instance of collections.OrderedDict mapping element names to their
    types (or things that are convertible to types).


#### Raises:

* <b>`TypeError`</b>: if the arguments are of the wrong types.
* <b>`ValueError`</b>: if the named tuple contains no elements.



## Properties

<h3 id="elements"><code>elements</code></h3>





## Methods

<h3 id="is_assignable_from"><code>is_assignable_from</code></h3>

``` python
is_assignable_from(other)
```





