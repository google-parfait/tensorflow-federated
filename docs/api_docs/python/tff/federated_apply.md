<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_apply" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_apply

```python
tff.federated_apply(
    func,
    arg
)
```

Applies a given function to a federated value on the `SERVER`.

#### Args:

*   <b>`func`</b>: A function to apply to the member content of `arg` on the
    `SERVER`. The parameter of this function must be of the same type as the
    member constituent of `arg`.
*   <b>`arg`</b>: A value of a TFF federated type placed at the `SERVER`, and
    with the `all_equal` bit set.

#### Returns:

A federated value on the `SERVER` that represents the result of applying `func`
to the member constituent of `arg`.

#### Raises:

*   <b>`TypeError`</b>: if the arguments are not of the appropriates
    computation_types.
