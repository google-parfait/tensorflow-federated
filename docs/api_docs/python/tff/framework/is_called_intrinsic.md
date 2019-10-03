<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.is_called_intrinsic" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.is_called_intrinsic

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_block_analysis.py">View
source</a>

Tests if `comp` is a called intrinsic with the given `uri`.

```python
tff.framework.is_called_intrinsic(
    comp,
    uri=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`comp`</b>: The computation building block to test.
*   <b>`uri`</b>: A uri or a collection of uris; the same as what is accepted by
    isinstance.

#### Returns:

`True` if `comp` is a called intrinsic with the given `uri`, otherwise `False`.
