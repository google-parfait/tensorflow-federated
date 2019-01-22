<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_map" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_map

``` python
tff.federated_map(
    mapping_fn,
    value
)
```



Defined in [`core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

Maps a federated value on CLIENTS pointwise using a given mapping function.

#### Args:

* <b>`mapping_fn`</b>: A mapping function to apply pointwise to member constituents of
    `value` on each of the participants in `CLIENTS`. The parameter of this
    function must be of the same type as the member constituents of `value`.
* <b>`value`</b>: A value of a TFF federated type placed at the `CLIENTS`, or a value
    that can be implicitly converted into a TFF federated type, e.g., by
    zipping.


#### Returns:

A federated value on `CLIENTS` that represents the result of mapping.


#### Raises:

* <b>`TypeError`</b>: If the arguments are not of the appropriate types.