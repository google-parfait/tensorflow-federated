<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.is_called_intrinsic" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.is_called_intrinsic

Returns `True` if `comp` is a called intrinsic with the `uri` or `uri`s.

```python
tff.framework.is_called_intrinsic(
    comp,
    uri=None
)
```

Defined in
[`python/core/impl/transformations.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py).

<!-- Placeholder for "Used in" -->

          Call
         /

Intrinsic

#### Args:

*   <b>`comp`</b>: The computation building block to test.
*   <b>`uri`</b>: A uri or a list, tuple, or set of uri.
