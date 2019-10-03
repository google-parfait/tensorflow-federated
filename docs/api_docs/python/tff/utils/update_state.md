<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.update_state" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.update_state

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/computation_utils.py">View
source</a>

Returns a new `state` with new values for fields in `kwargs`.

```python
tff.utils.update_state(
    state,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`state`</b>: the structure with named fields to update.
*   <b>`**kwargs`</b>: the list of key-value pairs of fields to update in
    `state`.

#### Raises:

*   <b>`KeyError`</b>: if kwargs contains a field that is not in state.
*   <b>`TypeError`</b>: if state is not a structure with named fields.
