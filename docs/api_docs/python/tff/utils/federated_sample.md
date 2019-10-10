<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.federated_sample" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.federated_sample

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/federated_aggregations.py">View
source</a>

Aggregation to produce uniform sample of at most `max_num_samples` values.

```python
tff.utils.federated_sample(
    value,
    max_num_samples=100
)
```

<!-- Placeholder for "Used in" -->

Each client value is assigned a random number when it is examined during each
accumulation. Each accumulate and merge only keeps the top N values based on the
random number. Report drops the random numbers and only returns the at most N
values sampled from the accumulated client values using standard reservoir
sampling (https://en.wikipedia.org/wiki/Reservoir_sampling), where N is user
provided `max_num_samples`.

#### Args:

*   <b>`value`</b>: A <a href="../../tff/Value.md"><code>tff.Value</code></a>
    placed on the <a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.
*   <b>`max_num_samples`</b>: The maximum number of samples to collect from
    client values. If fewer clients than the defined max sample size
    participated in the round of computation, the actual number of samples will
    equal the number of clients in the round.

#### Returns:

At most `max_num_samples` samples of the value from the
<a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.
