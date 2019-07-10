<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.federated_max" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.federated_max

Aggregation to find the maximum value from the
<a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

```python
tff.utils.federated_max(value)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/federated_aggregations.py>View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`value`</b>: A <a href="../../tff/Value.md"><code>tff.Value</code></a>
    placed on the <a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

#### Returns:

In the degenerate scenario that the `value` is aggregated over an empty set of
<a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>, the tensor
constituents of the result are set to the minimum of the underlying numeric data
type.
