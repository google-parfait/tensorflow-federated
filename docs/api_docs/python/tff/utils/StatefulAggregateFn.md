<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.StatefulAggregateFn" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
</div>

# tff.utils.StatefulAggregateFn

## Class `StatefulAggregateFn`

A simple container for a stateful aggregation function.

Defined in
[`core/utils/computation_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/computation_utils.py).

<!-- Placeholder for "Used in" -->

A typical (though trivial) example would be:

```
stateless_federated_mean = tff.utils.StatefulAggregateFn(
    initialize_fn=lambda: (),  # The state is an empty tuple.
    next_fn=lambda state, value, weight=None: (
        state, tff.federated_mean(value, weight=weight)))
```

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    initialize_fn,
    next_fn
)
```

Creates the StatefulFn.

#### Args:

*   <b>`initialize_fn`</b>: A no-arg function that returns a Python container
    which can be converted to a
    <a href="../../tff/Value.md"><code>tff.Value</code></a>, placed on the
    <a href="../../tff.md#SERVER"><code>tff.SERVER</code></a>, and passed as the
    first argument of `__call__`. This may be called in vanilla TensorFlow code,
    typically wrapped as a `tff.tf_compuatation`, as part of the initialization
    of a larger state object.
*   <b>`next_fn`</b>: A function matching the signature of `__call__`, see
    below.

## Methods

<h3 id="__call__"><code>__call__</code></h3>

```python
__call__(
    state,
    value,
    weight=None
)
```

Performs an aggregate of value@CLIENTS, with optional weight@CLIENTS.

This is a function intended to (only) be invoked in the context of a
<a href="../../tff/federated_computation.md"><code>tff.federated_computation</code></a>.
It shold be compatible with the TFF type signature

```
(state@SERVER, value@CLIENTS, weight@CLIENTS) ->
     (state@SERVER, aggregate@SERVER).
```

#### Args:

*   <b>`state`</b>: A <a href="../../tff/Value.md"><code>tff.Value</code></a>
    placed on the <a href="../../tff.md#SERVER"><code>tff.SERVER</code></a>.
*   <b>`value`</b>: A <a href="../../tff/Value.md"><code>tff.Value</code></a> to
    be aggregated, placed on the
    <a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.
*   <b>`weight`</b>: An optional
    <a href="../../tff/Value.md"><code>tff.Value</code></a> for weighting
    values, placed on the
    <a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

#### Returns:

A tuple of <a href="../../tff/Value.md"><code>tff.Value</code></a>s
(state@SERVER, aggregate@SERVER) where * state: The updated state. * aggregate:
The result of the aggregation of `value` weighted by `weight.

<h3 id="initialize"><code>initialize</code></h3>

```python
initialize()
```

Returns the initial state.
