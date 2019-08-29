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
tff.framework.create_local_executor(num_clients=None)
```

<!-- Placeholder for "Used in" -->

NOTE: This function is only available in Python 3.

#### Args:

*   <b>`num_clients`</b>: The number of clients. If specified, the executor
    factory function returned by `create_local_executor` will be configured to
    have exactly `num_clients` clients. If unspecified (`None`), then the
    function returned will attempt to infer cardinalities of all placements for
    which it is passed values.

#### Returns:

An executor factory function which returns a
<a href="../../tff/framework/Executor.md"><code>tff.framework.Executor</code></a>
upon invocation with a dict mapping placements to positive integers.

#### Raises:

*   <b>`ValueError`</b>: If the number of clients is specified and not one or
    larger.
