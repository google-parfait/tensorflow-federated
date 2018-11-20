<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.FederatedType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="all_equal"/>
<meta itemprop="property" content="member"/>
<meta itemprop="property" content="placement"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="is_assignable_from"/>
</div>

# tff.FederatedType

## Class `FederatedType`

Inherits From: [`Type`](../tff/Type.md)

An implementation of `Type` for representing federated types in TFF.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    member,
    placement,
    all_equal=False
)
```

Constructs a new federated type instance.

#### Args:

* <b>`member`</b>: An instance of `Type` (or something convertible to it) that
    represents the type of the member components of each value of this
    federated type.
* <b>`placement`</b>: The specification of placement that the member components
    of this federated type are hosted on. Must be either a placement
    literal such as `SERVER` or `CLIENTS` to refer to a globally defined
    placement, or a placement label to refer to a placement defined in
    other parts of a type signature. Specifying placement labels is not
    implemented yet.
* <b>`all_equal`</b>: A `bool` value that indicates whether all members of the
    federated type are equal (`True`), or are allowed to differ (`False`).



## Properties

<h3 id="all_equal"><code>all_equal</code></h3>



<h3 id="member"><code>member</code></h3>



<h3 id="placement"><code>placement</code></h3>





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





