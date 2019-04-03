<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.ModelWeights" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="non_trainable"/>
<meta itemprop="property" content="keras_weights"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="assign_weights_to"/>
<meta itemprop="property" content="from_model"/>
<meta itemprop="property" content="from_tff_value"/>
</div>

# tff.learning.framework.ModelWeights

## Class `ModelWeights`

Defined in
[`learning/model_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model_utils.py).

<!-- Placeholder for "Used in" -->

A container for the trainable and non-trainable variables of a `Model`.

Note this does not include the model's local variables.

It may also be used to hold other values that are parallel to these variables,
e.g., tensors corresponding to variable values, or updates to model variables.

<h2 id="__new__"><code>__new__</code></h2>

```python
@staticmethod
__new__(
    cls,
    trainable,
    non_trainable
)
```

## Properties

<h3 id="trainable"><code>trainable</code></h3>

<h3 id="non_trainable"><code>non_trainable</code></h3>

<h3 id="keras_weights"><code>keras_weights</code></h3>

Returns a list of weights in the same order as `tf.keras.Model.weights`.

(Assuming that this ModelWeights object corresponds to the weights of a keras
model).

IMPORTANT: this is not the same order as `tf.keras.Model.get_weights()`, and
hence will not work with `tf.keras.Model.set_weights()`. Instead, use
`tff.learning.ModelWeights.assign_weights_to`.

## Methods

<h3 id="assign_weights_to"><code>assign_weights_to</code></h3>

```python
assign_weights_to(keras_model)
```

Assign these TFF model weights to the weights of a `tf.keras.Model`.

#### Args:

*   <b>`keras_model`</b>: the `tf.keras.Model` object to assign weights to.

<h3 id="from_model"><code>from_model</code></h3>

```python
@classmethod
from_model(
    cls,
    model
)
```

<h3 id="from_tff_value"><code>from_tff_value</code></h3>

```python
@classmethod
from_tff_value(
    cls,
    anon_tuple
)
```
