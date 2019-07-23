<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.create_local_executor" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.create_local_executor

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_stacks.py">View
source</a>

Constructs an executor to execute computations on the local machine.

```python
tff.framework.create_local_executor(num_clients)
```

### Used in the tutorials:

*   [Federated Learning for Image Classification](https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification)
*   [High-performance simulations with TFF](https://www.tensorflow.org/federated/tutorials/simulations)

The initial temporary implementation requires that the number of clients be
specified in advance. This limitation will be removed in the near future.

NOTE: This function is only available in Python 3.

#### Args:

*   <b>`num_clients`</b>: The number of clients.

#### Returns:

An instance of
<a href="../../tff/framework/Executor.md"><code>tff.framework.Executor</code></a>
for single-machine use only.
