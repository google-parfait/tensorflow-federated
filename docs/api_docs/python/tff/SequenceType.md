<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.SequenceType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="element"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="is_assignable_from"/>
</div>

# tff.SequenceType

## Class `SequenceType`

Inherits From: [`Type`](../tff/Type.md)

An implementation of Type for representing types of sequences in TFF.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(element)
```

Constructs a new instance from the given element type.

#### Args:

* <b>`element`</b>: A specification of the element type, either an instance of Type
    or something convertible to it by to_type().



## Properties

<h3 id="element"><code>element</code></h3>





## Methods

<h3 id="is_assignable_from"><code>is_assignable_from</code></h3>

``` python
is_assignable_from(other)
```





