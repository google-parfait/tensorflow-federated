<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.ExecutorService" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="Compute"/>
<meta itemprop="property" content="CreateCall"/>
<meta itemprop="property" content="CreateSelection"/>
<meta itemprop="property" content="CreateTuple"/>
<meta itemprop="property" content="CreateValue"/>
<meta itemprop="property" content="Dispose"/>
<meta itemprop="property" content="Execute"/>
<meta itemprop="property" content="__init__"/>
</div>

# tff.framework.ExecutorService

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_service.py">View
source</a>

## Class `ExecutorService`

A wrapper around a target executor that makes it into a gRPC service.

<!-- Placeholder for "Used in" -->

NOTE: This component is only available in Python 3.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_service.py">View
source</a>

```python
__init__(
    executor,
    *args,
    **kwargs
)
```

Initialize self. See help(type(self)) for accurate signature.

## Methods

<h3 id="Compute"><code>Compute</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_service.py">View
source</a>

```python
Compute(
    request,
    context
)
```

Computes a value embedded in the executor.

#### Args:

*   <b>`request`</b>: An instance of `executor_pb2.ComputeRequest`.
*   <b>`context`</b>: An instance of `grpc.ServicerContext`.

#### Returns:

An instance of `executor_pb2.ComputeResponse`.

<h3 id="CreateCall"><code>CreateCall</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_service.py">View
source</a>

```python
CreateCall(
    request,
    context
)
```

Creates a call embedded in the executor.

#### Args:

*   <b>`request`</b>: An instance of `executor_pb2.CreateCallRequest`.
*   <b>`context`</b>: An instance of `grpc.ServicerContext`.

#### Returns:

An instance of `executor_pb2.CreateCallResponse`.

<h3 id="CreateSelection"><code>CreateSelection</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_service.py">View
source</a>

```python
CreateSelection(
    request,
    context
)
```

Creates a selection embedded in the executor.

#### Args:

*   <b>`request`</b>: An instance of `executor_pb2.CreateSelectionRequest`.
*   <b>`context`</b>: An instance of `grpc.ServicerContext`.

#### Returns:

An instance of `executor_pb2.CreateSelectionResponse`.

<h3 id="CreateTuple"><code>CreateTuple</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_service.py">View
source</a>

```python
CreateTuple(
    request,
    context
)
```

Creates a tuple embedded in the executor.

#### Args:

*   <b>`request`</b>: An instance of `executor_pb2.CreateTupleRequest`.
*   <b>`context`</b>: An instance of `grpc.ServicerContext`.

#### Returns:

An instance of `executor_pb2.CreateTupleResponse`.

<h3 id="CreateValue"><code>CreateValue</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_service.py">View
source</a>

```python
CreateValue(
    request,
    context
)
```

Creates a value embedded in the executor.

#### Args:

*   <b>`request`</b>: An instance of `executor_pb2.CreateValueRequest`.
*   <b>`context`</b>: An instance of `grpc.ServicerContext`.

#### Returns:

An instance of `executor_pb2.CreateValueResponse`.

<h3 id="Dispose"><code>Dispose</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/proto/v0/executor_pb2_grpc.py">View
source</a>

```python
Dispose(
    request,
    context
)
```

<h3 id="Execute"><code>Execute</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_service.py">View
source</a>

```python
Execute(
    request_iter,
    context
)
```
