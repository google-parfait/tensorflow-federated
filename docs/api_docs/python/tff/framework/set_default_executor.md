<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.set_default_executor" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.set_default_executor

Places an `executor`-backed execution context at the top of the stack.

```python
tff.framework.set_default_executor(executor=None)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/set_default_executor.py>View
source</a>

<!-- Placeholder for "Used in" -->

NOTE: This function is only available in Python 3.

#### Args:

*   <b>`executor`</b>: Either an instance of `executor_base.Executor`, or `None`
    which causes the default reference executor to be installed (as is the
    default).
