<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_map" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_map

``` python
tff.federated_map(
    value,
    mapping_fn
)
```

Maps a federated value on CLIENTS pointwise using a given mapping function.

#### Args:

* <b>`value`</b>: A value of a TFF federated type placed at the `CLIENTS`, or a value
    that can be implicitly converted into a TFF federated type, e.g., by
    zipping.
* <b>`mapping_fn`</b>: A mapping function to apply pointwise to member constituents of
    `value` on each of the participants in `CLIENTS`. The parameter of this
    function must be of the same type as the member constituents of `value`.


#### Returns:

A federated value on `CLIENTS` that represents the result of mapping.


#### Raises:

* <b>`TypeError`</b>: if the arguments are not of the appropriates computation_types.