# Tracing

[TOC]

Tracing the the process of constructing a [AST](compilation.md#ast) from a
Python function.

TODO(b/153500547): Describe and link the individual components of the tracing
system.

## Tracing a Federated Computation

At a high level, there are three components to tracing a Federated computation.

### Packing the arguments

Internally, a TFF computation only ever have zero or one argument. The arguments
provided to the
[computations.federated_computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/api/computations.py)
decorator describe type signature of the arguments to the TFF computation. TFF
uses this information to to determine how to pack the arguments of the Python
function into a single
[structure.Struct](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/common_libs/structure.py).
This argument is used to construct a new Python function which declares exactly
0 or 1 parameters, often referred to as `zero_or_one_arg_fn`.

Note: Using `Struct` as a single data structure to represent both Python `args`
and `kwargs` is the reason that `Struct` accepts both named and unnamed fields.

See
[function_utils.wrap_as_zero_or_one_arg_callable](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/utils/function_utils.py)
for more information.

### Tracing the function

This new zero-or-one-argument Python function is called, using a
[value_base.Value](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/api/value_base.py)
as a stand-in replacement for each argument. `Value` attempts to emulate the
behavior of the original argument type by implementing common Python dunder
methods (e.g. `__getattr__`).

In more detail, when there is exactly one argument, tracing is accomplished by:

1.  Constructing a
    [value_impl.ValueImpl](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/value_impl.py)
    backed by a
    [building_blocks.Reference](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
    with appropriate type signature to represent the argument.

2.  Invoking the function on the `ValueImpl`. This causes the Python runtime to
    invoke the dunder methods implemented by `ValueImpl`, which translates those
    dunder methods as AST construction. Each dunder method constructs a AST and
    returns a `ValueImpl` backed by that AST.

For example:

```python
def foo(x):
  return x[0]
```

Here the function’s parameter is a tuple and in the body of the fuction the 0th
element is selected. This invokes Python’s `__getitem__` method, which is
overridden on `ValueImpl`. In the simplest case, the implementation of
`ValueImpl.__getitem__` constructs a
[building_blocks.Selection](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
to represent the invocation of `__getitem__` and returns a `ValueImpl` backed by
this new `Selection`.

Tracing continues because each dunder methods return a `ValueImpl`, stamping out
every operation in the body of the function which causes one of the overriden
dunder methods to be invoked.

### Constructing the AST

The result of tracing the function is packaged into a
[building_blocks.Lambda](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
whose `parameter_name` and `parameter_type` map to the
[building_block.Reference](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
created to represent the packed arguments. The resulting `Lambda` is then
returned as a Python object that fully represents the user’s Python function.

## Tracing a TensorFlow Computation

TODO(b/153500547): Describe the process of tracing a TensorFlow compuation.
