<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.BatchOutput" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="loss"/>
<meta itemprop="property" content="predictions"/>
</div>

# tff.learning.BatchOutput

## Class `BatchOutput`

A structure that holds the output of a
<a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a>.

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model.py">View
source</a>

<!-- Placeholder for "Used in" -->

NOTE: All fields are optional (may be None).

-   `loss`: The scalar mean loss on the examples in the batch. If the model has
    multiple losses, it is the sum of all the individual losses.
-   `predictions`: Tensor of predictions on the examples. The first dimension
    must be the same size (the size of the batch).

## Properties

<h3 id="loss"><code>loss</code></h3>

<h3 id="predictions"><code>predictions</code></h3>
