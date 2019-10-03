<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.federated_min" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.federated_min

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/federated_aggregations.py">View
source</a>

Aggregation to find the minimum value from the
<a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

```python
tff.utils.federated_min(value)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`value`</b>: A <a href="../../tff/Value.md"><code>tff.Value</code></a>
    placed on the <a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

#### Returns:

In the degenerate scenario that the `value` is aggregated over an empty set of
<a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>, the tensor
constituents of the result are set to the maximum of the underlying numeric data
type.
