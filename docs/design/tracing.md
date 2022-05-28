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
[federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py)
decorator describe type signature of the arguments to the TFF computation. TFF
uses this information to to determine how to pack the arguments of the Python
function into a single
[structure.Struct](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/common_libs/structure.py).

Note: Using `Struct` as a single data structure to represent both Python `args`
and `kwargs` is the reason that `Struct` accepts both named and unnamed fields.

See
[function_utils.create_argument_unpacking_fn](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/computation/function_utils.py)
for more information.

### Tracing the function

When tracing a `federated_computation`, the user's function is called using
[value_impl.Value](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py)
as a stand-in replacement for each argument. `Value` attempts to emulate the
behavior of the original argument type by implementing common Python dunder
methods (e.g. `__getattr__`).

In more detail, when there is exactly one argument, tracing is accomplished by:

1.  Constructing a
    [value_impl.ValueImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py)
    backed by a
    [building_blocks.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
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
[building_blocks.Selection](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
to represent the invocation of `__getitem__` and returns a `ValueImpl` backed by
this new `Selection`.

Tracing continues because each dunder methods return a `ValueImpl`, stamping out
every operation in the body of the function which causes one of the overriden
dunder methods to be invoked.

### Constructing the AST

The result of tracing the function is packaged into a
[building_blocks.Lambda](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
whose `parameter_name` and `parameter_type` map to the
[building_block.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)
created to represent the packed arguments. The resulting `Lambda` is then
returned as a Python object that fully represents the user’s Python function.

## Tracing a TensorFlow Computation

TODO(b/153500547): Describe the process of tracing a TensorFlow compuation.

## Clean Error Messages from Exceptions During Tracing

At one point in TFF's history, the process of tracing the user's computation
involved passing through a number of wrapper functions before calling into the
user's function. This had the undesirable effect of producing error messages
like this:

```
Traceback (most recent call last):
  File "<user code>.py", in user_function
    @tff.federated_computation(...)
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<user code>", in user_function
    <some line of user code inside the federated_computation>
  File "<tff code>.py", tff_function
  ...
  File "<tff code>.py", tff_function
    <raise some error about something the user did wrong>
FederatedComputationWrapperTest.test_stackframes_in_errors.<locals>.DummyError
```

It's quite hard to find the bottom line of user code (the line that actually
contained the bug) in this traceback. This resulted in users reporting these
issues as TFF bugs and generally made users' lives more difficult.

Today, TFF goes to some trouble to ensure that these call stacks are free of
extra TFF functions. This is the reason for the use of generators in TFF's
tracing code, often in patterns that look like this:

```
# Instead of writing this:
def foo(fn, x):
  return 5 + fn(x + 1)

print(foo(user_fn, 20))

# TFF uses this pattern for its tracing code:
def foo(x):
  result = yield x + 1
  yield result + 5

fooer = foo(20)
arg = next(fooer)
result = fooer.send(user_fn(arg))
print(result)
```

This pattern allows the user's code (`user_fn` above) to be called at the top
level of the call stack while also allowing its arguments, output, and even
thread-local context to be manipulated by wrapping functions.

Some simple versions of this pattern can more simply be replaced by "before" and
"after" functions. For example, `foo` above could be replaced with:

```
def foo_before(x):
  return x + 1

def foo_after(x):
  return x + 5
```

This pattern should be preferred for cases that do not require shared state
between the "before" and "after" portions. However, more complex cases involving
complex state or context managers can be cumbersome to express this way:

```
# With the `yield` pattern:
def in_ctx(fn):
  with create_ctx():
    yield
    ... something in the context ...
  ...something after the context...
  yield

# WIth the `before` and `after` pattern:
def before():
  new_ctx = create_ctx()
  new_ctx.__enter__()
  return new_ctx

def after(ctx):
  ...something in the context...
  ctx.__exit__()
  ...something after the context...
```

It's much less clear in the latter example which code is running inside a
context, and the situation gets even less clear when more bits of state are
shared across the before and after sections.

Several other solutions to the general problem of "hide TFF functions from user
error messages" were attempted, including catching and reraising exceptions
(failed due to the inability to create an exception whose stack included only
the lowest level of user code without also including the code that called it),
catching exceptions and replacing their traceback with a filtered one (which is
CPython-specific and unsupported by the Python language), and replacing the
exception handler (fails because `sys.excepthook` isn't used by `absltest` and
is overriden by other frameworks). In the end, the generator-based
inversion-of-control allowed for the best end-user experience at the cost of
some TFF implementation complexity.
