<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.ModelWeights" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="non_trainable"/>
<meta itemprop="property" content="assign_weights_to"/>
<meta itemprop="property" content="from_model"/>
<meta itemprop="property" content="from_tff_value"/>
</div>

# tff.learning.framework.ModelWeights

## Class `ModelWeights`

A container for the trainable and non-trainable variables of a `Model`.

Defined in
[`python/learning/model_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model_utils.py).

<!-- Placeholder for "Used in" -->

Note this does not include the model's local variables.

It may also be used to hold other values that are parallel to these variables,
e.g., tensors corresponding to variable values, or updates to model variables.

## Properties

<h3 id="trainable"><code>trainable</code></h3>

<h3 id="non_trainable"><code>non_trainable</code></h3>

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
