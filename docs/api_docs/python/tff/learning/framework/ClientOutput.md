<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.ClientOutput" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="weights_delta"/>
<meta itemprop="property" content="weights_delta_weight"/>
<meta itemprop="property" content="model_output"/>
<meta itemprop="property" content="optimizer_output"/>
<meta itemprop="property" content="__new__"/>
</div>

# tff.learning.framework.ClientOutput

## Class `ClientOutput`

Structure for outputs returned from clients during federated optimization.

Defined in
[`learning/framework/optimizer_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/framework/optimizer_utils.py).

<!-- Placeholder for "Used in" -->

#### Fields:

-   `weights_delta`: a dictionary of updates to the model's trainable variables.
-   `weights_delta_weight`: weight to use in a weighted mean when aggregating
    `weights_delta`.
-   `model_output`: a structure matching
    <a href="../../../tff/learning/Model.md#report_local_outputs"><code>tff.learning.Model.report_local_outputs</code></a>,
    reflecting the results of training on the input dataset.
-   `optimizer_output`: additional metrics or other outputs defined by the
    optimizer.

<h2 id="__new__"><code>__new__</code></h2>

```python
__new__(
    _cls,
    weights_delta,
    weights_delta_weight,
    model_output,
    optimizer_output
)
```

Create new instance of ClientOutput(weights_delta, weights_delta_weight,
model_output, optimizer_output)

## Properties

<h3 id="weights_delta"><code>weights_delta</code></h3>

<h3 id="weights_delta_weight"><code>weights_delta_weight</code></h3>

<h3 id="model_output"><code>model_output</code></h3>

<h3 id="optimizer_output"><code>optimizer_output</code></h3>
