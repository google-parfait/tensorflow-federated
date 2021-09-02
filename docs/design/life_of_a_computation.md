# Life of a Computation

[TOC]

## Executing a Python function in TFF

This example is meant to highlight how a Python function becomes a TFF
computation and how the computation is evaluated by TFF.

**From a users perspective:**

```python
tff.backends.native.set_local_python_execution_context()  # 3

@tff.tf_computation(tf.int32)  # 2
def add_one(x):  # 1
  return x + 1

result = add_one(2) # 4
```

1.  Write a *Python* function.

1.  Decorate the *Python* function with `@tff.tf_computation`.

    Note: For now, it is only important that the Python function is decorated
    not the specifics of the decorator itself; this is explained in more detail
    [below](#tf-vs-tff-vs-python).

1.  Set a TFF [context](context.md).

1.  Invoke the *Python* function.

**From a TFF perspective:**

When the Python is **parsed**, the `@tff.tf_computation` decorator will
[trace](tracing.md) the Python function and construct a TFF computation.

When the decorated Python function is **invoked**, it is the TFF computation
which is invoked and TFF will [compile](compilation.md) and
[execute](execution.md) the computation in the [context](context.md) that was
set.

## TF vs TFF vs Python

```python
tff.backends.native.set_local_python_execution_context()

@tff.tf_computation(tf.int32)
def add_one(x):
  return x + 1

@tff.federated_computation(tff.type_at_clients(tf.int32))
def add_one_to_all_clients(values):
  return tff.federated_map(add_one, values)

values = [1, 2, 3]
values = add_one_to_all_clients(values)
values = add_one_to_all_clients(values)
>>> [3, 4, 5]
```

TODO(b/153500547): Describe TF vs TFF vs Python example.
