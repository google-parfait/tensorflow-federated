<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.EnhancedTrainableModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="federated_output_computation"/>
<meta itemprop="property" content="input_spec"/>
<meta itemprop="property" content="local_variables"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="forward_pass"/>
<meta itemprop="property" content="report_local_outputs"/>
<meta itemprop="property" content="train_on_batch"/>
</div>

# tff.learning.framework.EnhancedTrainableModel

## Class `EnhancedTrainableModel`

Inherits From:
[`EnhancedModel`](../../../tff/learning/framework/EnhancedModel.md),
[`TrainableModel`](../../../tff/learning/TrainableModel.md)

Defined in
[`learning/model_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model_utils.py).

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(model)
```

## Properties

<h3 id="federated_output_computation"><code>federated_output_computation</code></h3>

<h3 id="input_spec"><code>input_spec</code></h3>

<h3 id="local_variables"><code>local_variables</code></h3>

<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>

<h3 id="trainable_variables"><code>trainable_variables</code></h3>

<h3 id="weights"><code>weights</code></h3>

Returns a `tff.learning.ModelWeights`.

## Methods

<h3 id="forward_pass"><code>forward_pass</code></h3>

```python
forward_pass(
    batch_input,
    training=True
)
```

<h3 id="report_local_outputs"><code>report_local_outputs</code></h3>

```python
report_local_outputs()
```

<h3 id="train_on_batch"><code>train_on_batch</code></h3>

```python
train_on_batch(batch_input)
```
