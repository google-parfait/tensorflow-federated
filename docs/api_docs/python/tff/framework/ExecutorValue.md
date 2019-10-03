<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.ExecutorValue" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="compute"/>
</div>

# tff.framework.ExecutorValue

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_value_base.py">View
source</a>

## Class `ExecutorValue`

Represents the abstract interface for values embedded within executors.

Inherits From: [`TypedObject`](../../tff/TypedObject.md)

<!-- Placeholder for "Used in" -->

The embedded values may represent computations in-flight that may materialize in
the future or fail before they materialize.

NOTE: This component is only available in Python 3.

## Properties

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

## Methods

<h3 id="compute"><code>compute</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/executor_value_base.py">View
source</a>

```python
compute()
```

A coroutine that asynchronously returns the computed form of the value.

The computed form of a value can take a number of forms, such as primitive types
in Python, numpy arrays, or even eager tensors in case this is an eager
executor, or an executor backed by an eager one.

#### Returns:

The computed form of the value, as defined above.
