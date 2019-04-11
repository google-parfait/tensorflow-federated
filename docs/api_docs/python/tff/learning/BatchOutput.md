<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.BatchOutput" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="loss"/>
<meta itemprop="property" content="predictions"/>
<meta itemprop="property" content="__new__"/>
</div>

# tff.learning.BatchOutput

## Class `BatchOutput`





Defined in [`learning/model.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model.py).

<!-- Placeholder for "Used in" -->

A structure that holds the output of a <a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a>.

NOTE: All fields are optional (may be None).

-   `loss`: The scalar mean loss on the examples in the batch.
-   `predictions`: Tensor of predictions on the examples. The first dimension
    must be the same size (the size of the batch).

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    loss,
    predictions
)
```

Create new instance of BatchOutput(loss, predictions)



## Properties

<h3 id="loss"><code>loss</code></h3>



<h3 id="predictions"><code>predictions</code></h3>





