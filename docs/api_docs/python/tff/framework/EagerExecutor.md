<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.EagerExecutor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_call"/>
<meta itemprop="property" content="create_selection"/>
<meta itemprop="property" content="create_tuple"/>
<meta itemprop="property" content="create_value"/>
</div>

# tff.framework.EagerExecutor

## Class `EagerExecutor`

The eager executor only runs TensorFlow, synchronously, in eager mode.

Inherits From: [`Executor`](../../tff/framework/Executor.md)

Defined in
[`python/core/impl/eager_executor.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/eager_executor.py).

<!-- Placeholder for "Used in" -->

TODO(b/134764569): Add support for data as a building block.

This executor understands the following TFF types: tensors, sequences, named
tuples, and functions. It does not understand placements, federated, or abstract
types.

This executor understands the following kinds of TFF computation building
blocks: tensorflow computations, and external data. It does not understand
lambda calculus or any compositional constructs. Tuples and selections can only
be created using `create_tuple()` and `create_selection()` in the API.

The arguments to be ingested can be Python constants of simple types, nested
structures of those, as well as eager tensors and eager datasets.

The external data references must identify files available in the executor's
filesystem. The exact format is yet to be documented.

The executor will be able to place work on specific devices (e.g., on GPUs). In
contrast to the reference executor, it handles data sets in a pipelined fashion,
and does not place limits on the data set sizes. It also avoids marshaling
TensorFlow values in and out between calls.

It does not deal with multithreading, checkpointing, federated computations, and
other concerns to be covered by separate executor components. It runs the
operations it supports in a synchronous fashion. Asynchrony and other aspects
not supported here should be handled by composing this executor with other
executors into a complex executor stack, rather than mixing in all the logic.

NOTE: This component is only available in Python 3.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(device=None)
```

Creates a new instance of an eager executor.

#### Args:

*   <b>`device`</b>: An optional name of the device that this executor will
    schedule all of its operations to run on. It is the caller's responsibility
    to select a correct device name. For example, the list of physical devices
    can be obtained using `tf.config.experimental.list_physical_devices()`.

#### Raises:

*   <b>`RuntimeError`</b>: If not executing eagerly.
*   <b>`TypeError`</b>: If the device name is not a string.
*   <b>`ValueError`</b>: If there is no device `device`.

## Methods

<h3 id="create_call"><code>create_call</code></h3>

```python
create_call(
    comp,
    arg=None
)
```

Creates a call to `comp` with optional `arg`.

#### Args:

*   <b>`comp`</b>: As documented in `executor_base.Executor`.
*   <b>`arg`</b>: As documented in `executor_base.Executor`.

#### Returns:

An instance of `EagerValue` representing the result of the call.

#### Raises:

*   <b>`RuntimeError`</b>: If not executing eagerly.
*   <b>`TypeError`</b>: If the arguments are of the wrong types.

<h3 id="create_selection"><code>create_selection</code></h3>

```python
create_selection(
    source,
    index=None,
    name=None
)
```

Creates a selection from `source`.

#### Args:

*   <b>`source`</b>: As documented in `executor_base.Executor`.
*   <b>`index`</b>: As documented in `executor_base.Executor`.
*   <b>`name`</b>: As documented in `executor_base.Executor`.

#### Returns:

An instance of `EagerValue` that represents the constructed selection.

#### Raises:

*   <b>`TypeError`</b>: If arguments are of the wrong types.
*   <b>`ValueError`</b>: If either both, or neither of `name` and `index` are
    present.

<h3 id="create_tuple"><code>create_tuple</code></h3>

```python
create_tuple(elements)
```

Creates a tuple of `elements`.

#### Args:

*   <b>`elements`</b>: As documented in `executor_base.Executor`.

#### Returns:

An instance of `EagerValue` that represents the constructed tuple.

<h3 id="create_value"><code>create_value</code></h3>

```python
create_value(
    value,
    type_spec=None
)
```

Embeds `value` of type `type_spec` within this executor.

#### Args:

*   <b>`value`</b>: An object that represents the value to embed within the
    executor.
*   <b>`type_spec`</b>: The
    <a href="../../tff/Type.md"><code>tff.Type</code></a> of the value
    represented by this object, or something convertible to it. Can optionally
    be `None` if `value` is an instance of `typed_object.TypedObject`.

#### Returns:

An instance of `EagerValue`.

#### Raises:

*   <b>`RuntimeError`</b>: If not executing eagerly.
*   <b>`TypeError`</b>: If the arguments are of the wrong types.
*   <b>`ValueError`</b>: If the type was not specified and cannot be determined
    from the value.
