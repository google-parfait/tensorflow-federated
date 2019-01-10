<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_value" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_value

```python
tff.federated_value(
    value,
    placement
)
```

Returns a federated value at `placement`, with `value` as the constituent.

#### Args:

*   <b>`value`</b>: A value of a non-federated TFF type to be placed.
*   <b>`placement`</b>: The desired result placement, either `SERVER` or
    `CLIENTS`.

#### Returns:

A federated value with the given placement `placement`, and the member
constituent `value` equal at all locations.

#### Raises:

*   <b>`TypeError`</b>: if the arguments are not of the appropriates
    computation_types.
