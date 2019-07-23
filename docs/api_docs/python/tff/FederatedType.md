<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.FederatedType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="all_equal"/>
<meta itemprop="property" content="member"/>
<meta itemprop="property" content="placement"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="compact_representation"/>
<meta itemprop="property" content="formatted_representation"/>
</div>

# tff.FederatedType

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

## Class `FederatedType`

An implementation of <a href="../tff/Type.md"><code>tff.Type</code></a>
representing federated types in TFF.

Inherits From: [`Type`](../tff/Type.md)

### Used in the tutorials:

*   [Custom Federated Algorithms, Part 1: Introduction to the Federated Core](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1)
*   [Custom Federated Algorithms, Part 2: Implementing Federated Averaging](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_2)

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
__init__(
    member,
    placement,
    all_equal=None
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
    federated type are equal (`True`), or are allowed to differ (`False`). If
    `all_equal` is `None`, the value is selected as the default for the
    placement, e.g., `True` for
    <a href="../tff.md#SERVER"><code>tff.SERVER</code></a> and `False` for
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

## Properties

<h3 id="all_equal"><code>all_equal</code></h3>

<h3 id="member"><code>member</code></h3>

<h3 id="placement"><code>placement</code></h3>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
__eq__(other)
```

Determines whether two type definitions are identical.

Note that this notion of equality is stronger than equivalence. Two types with
equivalent definitions may not be identical, e.g., if they represent templates
with differently named type variables in their definitions.

#### Args:

*   <b>`other`</b>: The other type to compare against.

#### Returns:

`True` iff type definitions are syntatically identical (as defined above), or
`False` otherwise.

#### Raises:

*   <b>`NotImplementedError`</b>: If not implemented in the derived class.

<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
__ne__(other)
```

Return self!=value.

<h3 id="compact_representation"><code>compact_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
compact_representation()
```

Returns the compact string representation of this type.

<h3 id="formatted_representation"><code>formatted_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

```python
formatted_representation()
```

Returns the formatted string representation of this type.
