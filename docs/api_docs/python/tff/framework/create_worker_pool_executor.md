<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.create_worker_pool_executor" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.create_worker_pool_executor

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_stacks.py">View
source</a>

Create an executor backed by a worker pool.

```python
tff.framework.create_worker_pool_executor(
    executors,
    max_fanout=100
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`executors`</b>: A list of
    <a href="../../tff/framework/Executor.md"><code>tff.framework.Executor</code></a>
    instances that forward work to workers in the worker pool. These can be any
    type of executors, but in most scenarios, they will be instances of
    <a href="../../tff/framework/RemoteExecutor.md"><code>tff.framework.RemoteExecutor</code></a>.
*   <b>`max_fanout`</b>: The maximum fanout at any point in the aggregation
    hierarchy. If `num_clients > max_fanout`, the constructed executor stack
    will consist of multiple levels of aggregators. The height of the stack will
    be on the order of `log(num_clients) / log(max_fanout)`.

#### Returns:

An instance of
<a href="../../tff/framework/Executor.md"><code>tff.framework.Executor</code></a>.
