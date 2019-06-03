<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.ClientDeltaFn" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__call__"/>
</div>

# tff.learning.framework.ClientDeltaFn

## Class `ClientDeltaFn`

Represents a client computation that produces an update to a model.

Defined in
[`python/learning/framework/optimizer_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/framework/optimizer_utils.py).

<!-- Placeholder for "Used in" -->

## Properties

<h3 id="variables"><code>variables</code></h3>

Returns all the variables of this object.

Note this only includes variables that are part of the state of this object, and
not the model variables themselves.

#### Returns:

An iterable of `tf.Variable` objects.

## Methods

<h3 id="__call__"><code>__call__</code></h3>

```python
__call__(
    dataset,
    initial_weights
)
```

Defines the complete client computation.

Typically implementations should be decorated with `tf.function`.

#### Args:

*   <b>`dataset`</b>: A `tf.data.Dataset` producing batches than can be fed to
    <a href="../../../tff/learning/Model.md#forward_pass"><code>tff.learning.Model.forward_pass</code></a>.
*   <b>`initial_weights`</b>: A dictionary of initial values for all trainable
    and non-trainable model variables, keyed by name. This will be supplied by
    the server in Federated Averaging.

#### Returns:

An `optimizer_utils.ClientOutput` namedtuple.
