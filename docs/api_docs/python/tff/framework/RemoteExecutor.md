<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.RemoteExecutor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_call"/>
<meta itemprop="property" content="create_selection"/>
<meta itemprop="property" content="create_tuple"/>
<meta itemprop="property" content="create_value"/>
</div>

# tff.framework.RemoteExecutor

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/remote_executor.py">View
source</a>

## Class `RemoteExecutor`

The remote executor is a local proxy for a remote executor instance.

Inherits From: [`Executor`](../../tff/framework/Executor.md)

<!-- Placeholder for "Used in" -->

NOTE: This component is only available in Python 3.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/remote_executor.py">View
source</a>

```python
__init__(
    channel,
    rpc_mode='REQUEST_REPLY',
    thread_pool_executor=None
)
```

Creates a remote executor.

#### Args:

*   <b>`channel`</b>: An instance of `grpc.Channel` to use for communication
    with the remote executor service.
*   <b>`rpc_mode`</b>: Optional mode of calling the remote executor. Must be
    either 'REQUEST_REPLY' or 'STREAMING' (defaults to 'REQUEST_REPLY'). This
    option will be removed after the request-reply interface is deprecated.
*   <b>`thread_pool_executor`</b>: Optional concurrent.futures.Executor used to
    wait for the reply to a streaming RPC message. Uses the default Executor if
    not specified.

## Methods

<h3 id="create_call"><code>create_call</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/remote_executor.py">View
source</a>

```python
create_call(
    comp,
    arg=None
)
```

A coroutine that creates a call to `comp` with optional argument `arg`.

#### Args:

*   <b>`comp`</b>: The computation to invoke. It must have been first embedded
    in the executor by calling `create_value()` on it first.
*   <b>`arg`</b>: An optional argument of the call, or `None` if no argument was
    supplied. If it is present, it must have been embedded in the executor by
    calling `create_value()` on it first.

#### Returns:

An instance of `executor_value_base.ExecutorValue` that represents the
constructed vall.

<h3 id="create_selection"><code>create_selection</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/remote_executor.py">View
source</a>

```python
create_selection(
    source,
    index=None,
    name=None
)
```

A coroutine that creates a selection from `source`.

#### Args:

*   <b>`source`</b>: The source to select from. The source must have been
    embedded in this executor by invoking `create_value()` on it first.
*   <b>`index`</b>: An optional integer index. Either this, or `name` must be
    present.
*   <b>`name`</b>: An optional string name. Either this, or `index` must be
    present.

#### Returns:

An instance of `executor_value_base.ExecutorValue` that represents the
constructed selection.

<h3 id="create_tuple"><code>create_tuple</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/remote_executor.py">View
source</a>

```python
create_tuple(elements)
```

A coroutine that creates a tuple of `elements`.

#### Args:

*   <b>`elements`</b>: An enumerable or dict with the elements to create a tuple
    from. The elements must all have been embedded in this executor by invoking
    `create_value()` on them first.

#### Returns:

An instance of `executor_value_base.ExecutorValue` that represents the
constructed tuple.

<h3 id="create_value"><code>create_value</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/remote_executor.py">View
source</a>

```python
create_value(
    value,
    type_spec=None
)
```

A coroutine that creates embedded value from `value` of type `type_spec`.

This function is used to embed a value within the executor. The argument can be
one of the plain Python types, a nested structure, a representation of a TFF
computation, etc. Once embedded, the value can be further passed around within
the executor. For functional values, embedding them prior to invocation
potentally allows the executor to amortize overhead across multiple calls.

#### Args:

*   <b>`value`</b>: An object that represents the value to embed within the
    executor.
*   <b>`type_spec`</b>: An optional
    <a href="../../tff/Type.md"><code>tff.Type</code></a> of the value
    represented by this object, or something convertible to it. The type can
    only be omitted if the value is a instance of
    <a href="../../tff/TypedObject.md"><code>tff.TypedObject</code></a>.

#### Returns:

An instance of `executor_value_base.ExecutorValue` that represents the embedded
value.
