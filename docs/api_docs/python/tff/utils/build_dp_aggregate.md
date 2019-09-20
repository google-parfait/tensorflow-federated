<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.build_dp_aggregate" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.build_dp_aggregate

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/differential_privacy.py">View
source</a>

Builds a stateful aggregator for tensorflow_privacy DPQueries.

```python
tff.utils.build_dp_aggregate(
    query,
    value_type_fn=_default_get_value_type_fn,
    from_anon_tuple_fn=_default_from_anon_tuple_fn
)
```

<!-- Placeholder for "Used in" -->

The returned StatefulAggregateFn can be called with any nested structure for the
values being statefully aggregated. However, it's necessary to provide two
functions as arguments which indicate the properties (the tff.Type and the
AnonymousTuple conversion) of the nested structure that will be used. If using
an OrderedDict as the value's nested structure, the defaults for the arguments
suffice.

#### Args:

*   <b>`query`</b>: A DPQuery to aggregate. For compatibility with
    tensorflow_federated, the global_state and sample_state of the query must be
    structures supported by tf.nest.
*   <b>`value_type_fn`</b>: Python function that takes the value argument of
    next_fn and returns the value type. This will be used in determining the
    TensorSpecs that establish the initial sample state. If the value being
    aggregated is an OrderedDict, the default for this argument can be used.
    This argument probably gets removed once b/123092620 is addressed (and the
    associated processing step gets replaced with a simple call to
    value.type_signature.member).
*   <b>`from_anon_tuple_fn`</b>: Python function that takes a client record and
    converts it to the container type that it was in before passing through TFF.
    (Right now, TFF computation causes the client record to be changed into an
    AnonymousTuple, and this method corrects for that). If the value being
    aggregated is an OrderedDict, the default for this argument can be used.
    This argument likely goes away once b/123092620 is addressed. The default
    behavior assumes that the client record (before being converted to
    AnonymousTuple) was an OrderedDict containing a flat structure of Tensors
    (as it is if using the tff.learning APIs like
    tff.learning.build_federated_averaging_process).

#### Returns:

A tuple of: - a `computation_utils.StatefulAggregateFn` that aggregates
according to the query - the TFF type of the DP aggregator's global state
