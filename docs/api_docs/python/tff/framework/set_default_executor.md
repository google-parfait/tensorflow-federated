<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.set_default_executor" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.set_default_executor

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/set_default_executor.py">View
source</a>

Places an `executor`-backed execution context at the top of the stack.

```python
tff.framework.set_default_executor(executor=None)
```

### Used in the tutorials:

*   [Federated Learning for Image Classification](https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification)
*   [High-performance simulations with TFF](https://www.tensorflow.org/federated/tutorials/simulations)

NOTE: This function is only available in Python 3.

#### Args:

*   <b>`executor`</b>: Either an instance of `executor_base.Executor`, or `None`
    which causes the default reference executor to be installed (as is the
    default).
