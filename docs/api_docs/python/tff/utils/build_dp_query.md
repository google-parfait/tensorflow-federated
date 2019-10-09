<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.build_dp_query" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.build_dp_query

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/differential_privacy.py">View
source</a>

Makes a `DPQuery` to estimate vector averages with differential privacy.

```python
tff.utils.build_dp_query(
    clip,
    noise_multiplier,
    expected_total_weight,
    adaptive_clip_learning_rate=0,
    target_unclipped_quantile=None,
    clipped_count_budget_allocation=None,
    expected_num_clients=None,
    use_per_vector=False,
    model=None
)
```

<!-- Placeholder for "Used in" -->

Supports many of the types of query available in tensorflow_privacy, including
nested ("per-vector") queries as described in
https://arxiv.org/pdf/1812.06210.pdf, and quantile-based adaptive clipping as
described in https://arxiv.org/abs/1905.03871.

#### Args:

*   <b>`clip`</b>: The query's L2 norm bound.
*   <b>`noise_multiplier`</b>: The ratio of the (effective) noise stddev to the
    clip.
*   <b>`expected_total_weight`</b>: The expected total weight of all clients,
    used as the denominator for the average computation.
*   <b>`adaptive_clip_learning_rate`</b>: Learning rate for quantile-based
    adaptive clipping. If 0, fixed clipping is used. If per-vector clipping is
    enabled, the learning rate of each vector is proportional to that vector's
    initial clip, such that the sum of all per-vector learning rates equals
    this.
*   <b>`target_unclipped_quantile`</b>: Target unclipped quantile for adaptive
    clipping.
*   <b>`clipped_count_budget_allocation`</b>: The fraction of privacy budget to
    use for estimating clipped counts.
*   <b>`expected_num_clients`</b>: The expected number of clients for estimating
    clipped fractions.
*   <b>`use_per_vector`</b>: If True, clip each weight tensor independently.
    Otherwise, global clipping is used. The clipping norm for each vector (or
    the initial clipping norm, in the case of adaptive clipping) is proportional
    to the sqrt of the vector dimensionality while the total bound still equals
    `clip`.
*   <b>`model`</b>: A
    <a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a> to
    determine the structure of model weights. Required only if use_per_vector is
    True.

#### Returns:

A `DPQuery` suitable for use in a call to `build_dp_aggregate` to perform
Federated Averaging with differential privacy.
