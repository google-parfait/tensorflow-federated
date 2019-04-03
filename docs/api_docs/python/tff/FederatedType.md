<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.FederatedType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="all_equal"/>
<meta itemprop="property" content="member"/>
<meta itemprop="property" content="placement"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tff.FederatedType

## Class `FederatedType`

Inherits From: [`Type`](../tff/Type.md)

Defined in
[`core/api/computation_types.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py).

<!-- Placeholder for "Used in" -->

An implementation of <a href="../tff/Type.md"><code>tff.Type</code></a>
representing federated types in TFF.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    member,
    placement,
    all_equal=False
)
```

Constructs a new federated type instance.

#### Args:

*   <b>`member`</b>: An instance of
    <a href="../tff/Type.md"><code>tff.Type</code></a> (or something convertible
    to it) that represents the type of the member components of each value of
    this federated type.
*   <b>`placement`</b>: The specification of placement that the member
    components of this federated type are hosted on. Must be either a placement
    literal such as <a href="../tff.md#SERVER"><code>tff.SERVER</code></a> or
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a> to refer to a
    globally defined placement, or a placement label to refer to a placement
    defined in other parts of a type signature. Specifying placement labels is
    not implemented yet.
*   <b>`all_equal`</b>: A `bool` value that indicates whether all members of the
    federated type are equal (`True`), or are allowed to differ (`False`).

## Properties

<h3 id="all_equal"><code>all_equal</code></h3>

<h3 id="member"><code>member</code></h3>

<h3 id="placement"><code>placement</code></h3>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

```python
__eq__(other)
```

<h3 id="__ne__"><code>__ne__</code></h3>

```python
__ne__(other)
```
