<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_zip" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_zip

```python
tff.federated_zip(value)
```

Defined in
[`core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

<!-- Placeholder for "Used in" -->

Converts an N-tuple of federated values into a federated N-tuple value.

#### Args:

*   <b>`value`</b>: A value of a TFF named tuple type, the elements of which are
    federated values with the same placement.

#### Returns:

A federated value placed at the same location as the members of `value`, in
which every member component is a named tuple that consists of the corresponding
member components of the elements of `value`.

#### Raises:

*   <b>`TypeError`</b>: if the argument is not a named tuple of federated values
    with the same placement.
