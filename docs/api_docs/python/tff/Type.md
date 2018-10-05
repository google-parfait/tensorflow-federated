<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.Type" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="is_assignable_from"/>
</div>

# tff.Type

## Class `Type`



An abstract interface for all classes that represent TFF types.

## Methods

<h3 id="is_assignable_from"><code>is_assignable_from</code></h3>

``` python
is_assignable_from(other)
```

Determines whether this TFF type is assignable from another TFF type.

#### Args:

* <b>`other`</b>: Another type, an instance of Type.


#### Returns:

True if self is assignable from other, False otherwise.



