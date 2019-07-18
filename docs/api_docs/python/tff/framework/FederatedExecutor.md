<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.FederatedExecutor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_call"/>
<meta itemprop="property" content="create_selection"/>
<meta itemprop="property" content="create_tuple"/>
<meta itemprop="property" content="create_value"/>
</div>

# tff.framework.FederatedExecutor

## Class `FederatedExecutor`

The federated executor orchestrates federated computations.

Inherits From: [`Executor`](../../tff/framework/Executor.md)

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/federated_executor.py">View
source</a>

<!-- Placeholder for "Used in" -->

NOTE: This component is only available in Python 3.

The intrinsics currently implemented include: - federated_aggregate -
federated_apply - federated_broadcast - federated_map - federated_mean -
federated_reduce - federated_sum - federated_value - federated_weighted_mean -
federated_zip

This executor is only responsible for handling federated types and federated
operators, and a delegation of work to an underlying collection of target
executors associated with individual system participants. This executor does not
interpret lambda calculus and compositional constructs (blocks, etc.). It
understands placements, selected intrinsics (federated operators), it can handle
tuples, selections, and calls in a limited way (to the extent that it deals with
intrinsics or lambda expressions it can delegate).

The initial implementation of the executor only supports the two basic types of
placements (SERVER and CLIENTS), and does not have a built-in concept of
intermediate aggregation, partitioning placements, clustering clients, etc.

The initial implementation also does not attempt at performing optimizations in
case when the constituents of this executor are either located on the same
machine (where marshaling/unmarshaling could be avoided), or when they have the
`all_equal` property (and a single value could be shared by them all).

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/federated_executor.py">View
source</a>

```python
__init__(target_executors)
```

Creates a federated executor backed by a collection of target executors.

#### Args:

*   <b>`target_executors`</b>: A dictionary mapping placements to executors or
    lists of executors associated with these placements. The keys in this
    dictionary can be either placement literals, or `None` to specify the
    executor for unplaced computations. The values can be either single
    executors (if there only is a single participant associated with that
    placement, as would typically be the case with
    <a href="../../tff.md#SERVER"><code>tff.SERVER</code></a>) or lists of
    target executors.

#### Raises:

*   <b>`ValueError`</b>: If the value is unrecognized (e.g., a nonexistent
    intrinsic).

## Methods

<h3 id="create_call"><code>create_call</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/federated_executor.py">View
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

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/federated_executor.py">View
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

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/federated_executor.py">View
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

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/federated_executor.py">View
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
