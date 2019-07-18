<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.utils

Utility classes/functions built on top of TensorFlow Federated Core API.

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/__init__.py">View
source</a>

<!-- Placeholder for "Used in" -->

All components that depend on utils should import symbols from this file rather
than directly importing individual modules. For this reason, the visibility for
the latter is set to private and should remain such. The code in utils must not
depend on implementation classes. It should be written against the Core API.

## Classes

[`class IterativeProcess`](../tff/utils/IterativeProcess.md): A process that
includes an initialization and iterated computation.

[`class StatefulAggregateFn`](../tff/utils/StatefulAggregateFn.md): A simple
container for a stateful aggregation function.

[`class StatefulBroadcastFn`](../tff/utils/StatefulBroadcastFn.md): A simple
container for a stateful broadcast function.

## Functions

[`assign(...)`](../tff/utils/assign.md): Creates an op that assigns `target`
from `source`.

[`federated_max(...)`](../tff/utils/federated_max.md): Aggregation to find the
maximum value from the <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

[`federated_min(...)`](../tff/utils/federated_min.md): Aggregation to find the
minimum value from the <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

[`get_variables(...)`](../tff/utils/get_variables.md): Creates a set of
variables that matches the given `type_spec`.

[`identity(...)`](../tff/utils/identity.md): Applies `tf.identity` pointwise to
`source`.

[`update_state(...)`](../tff/utils/update_state.md): Returns a new `state` with
new values for fields in `kwargs`.
