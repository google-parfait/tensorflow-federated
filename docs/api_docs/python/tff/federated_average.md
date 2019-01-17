<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_average" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_average

```python
tff.federated_average(
    value,
    weight=None
)
```

Defined in
[`core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

Computes a `SERVER` average of `value` placed on `CLIENTS`.

#### Args:

*   <b>`value`</b>: The value to be averaged. Must be of a TFF federated type
    placed at `CLIENTS`. The value may be structured, e.g., its member
    constituents can be named tuples. The tensor types that the value is
    composed of must be floating-point or complex.
*   <b>`weight`</b>: An optional weight, a TFF federated integer or
    floating-point tensor value, also placed at `CLIENTS`.

#### Returns:

A representation at the `SERVER` of an average of the member constituents of
`value`, optionally weighted with `weight` if specified (otherwise, the member
constituents contributed by all clients are equally weighted).

#### Raises:

*   <b>`TypeError`</b>: if `value` is not a federated TFF value placed at
    `CLIENTS`, or if `weight` is not a federated integer or a floating-point
    tensor with the matching placement.
