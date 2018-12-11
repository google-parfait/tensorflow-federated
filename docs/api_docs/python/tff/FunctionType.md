<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.FunctionType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="parameter"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="is_assignable_from"/>
</div>

# tff.FunctionType

## Class `FunctionType`

Inherits From: [`Type`](../tff/Type.md)

An implementation of `Type` for representing functional types in TFF.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    parameter,
    result
)
```

Constructs a new instance from the given parameter and result types.

#### Args:

* <b>`parameter`</b>: A specification of the parameter type, either an instance of
    `Type` or something convertible to it by `to_type()`.
* <b>`result`</b>: A specification of the result type, either an instance of `Type`
    or something convertible to it by `to_type()`.



## Properties

<h3 id="parameter"><code>parameter</code></h3>



<h3 id="result"><code>result</code></h3>





## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```



<h3 id="__ge__"><code>__ge__</code></h3>

``` python
__ge__(other)
```



<h3 id="__gt__"><code>__gt__</code></h3>

``` python
__gt__(other)
```



<h3 id="__le__"><code>__le__</code></h3>

``` python
__le__(other)
```



<h3 id="__lt__"><code>__lt__</code></h3>

``` python
__lt__(other)
```



<h3 id="__ne__"><code>__ne__</code></h3>

``` python
__ne__(other)
```



<h3 id="is_assignable_from"><code>is_assignable_from</code></h3>

``` python
is_assignable_from(other)
```





