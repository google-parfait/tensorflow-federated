<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.Type" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="is_assignable_from"/>
</div>

# tff.Type

## Class `Type`



An abstract interface for all classes that represent TFF types.

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



