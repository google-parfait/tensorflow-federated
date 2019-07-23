<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.StatefulBroadcastFn" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
</div>

# tff.utils.StatefulBroadcastFn

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/computation_utils.py">View
source</a>

## Class `StatefulBroadcastFn`

A simple container for a stateful broadcast function.

<!-- Placeholder for "Used in" -->

A typical (though trivial) example would be:

```
stateless_federated_broadcast = tff.utils.StatefulBroadcastFn(
  initialize_fn=lambda: (),
  next_fn=lambda state, value: (
      state, tff.federated_broadcast(value)))
```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/computation_utils.py">View
source</a>

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

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/computation_utils.py">View
source</a>

```python
__call__(
    state,
    value
)
```

Performs a broadcast of `value@SERVER`, producing `value@CLIENTS`.

This is a function intended to (only) be invoked in the context of a
<a href="../../tff/federated_computation.md"><code>tff.federated_computation</code></a>.
It shold be compatible with the TFF type signature `(state@SERVER, value@SERVER)
-> (state@SERVER, value@CLIENTS)`.

#### Args:

*   <b>`state`</b>: A <a href="../../tff/Value.md"><code>tff.Value</code></a>
    placed on the <a href="../../tff.md#SERVER"><code>tff.SERVER</code></a>.
*   <b>`value`</b>: A <a href="../../tff/Value.md"><code>tff.Value</code></a>
    placed on the <a href="../../tff.md#SERVER"><code>tff.SERVER</code></a>, to
    be broadcast to the
    <a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

#### Returns:

A tuple of <a href="../../tff/Value.md"><code>tff.Value</code></a>s
`(state@SERVER, value@CLIENTS)` where

*   `state`: The updated state.
*   `value`: The input `value` now placed (communicated) to the
    <a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

<h3 id="initialize"><code>initialize</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/computation_utils.py">View
source</a>

```python
initialize()
```

Returns the initial state.
