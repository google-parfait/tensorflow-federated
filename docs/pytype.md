# Pytype

[Pytype](https://github.com/google/pytype) is a static analyzer for Python,
which checks and infers types for Python code.

## Benefits And Challenges

There are lots of advantages to using Pytype, see
https://github.com/google/pytype for more information. However, how type
annotations are interpreted by Pytype and the errors yielded by Pytype are
sometimes inconvenient for TensorFlow Federated readability.

*   Decorators

Pytype checks annotations against the function that they are annotating; and if
that function is decorated, a new function is created in which those same
annotations may not longer apply. Both TensorFlow and TensorFlow Federated use
decorators that dramatically transform the inputs and outputs of the decorated
function; meaning that functions decorated with `@tf.function`,
`@tff.tf_computation`, or `@tff.federated_computation` may behave surprisingly
when analyzed with pytype.

For example:

```
def decorator(fn):

  def wrapper():
    fn()
    return 10  # Anything decorated with this decorator will return a `10`.

  return wrapper


@decorator
def foo() -> str:
  return 'string'


@decorator
def bar() -> int:  # However, this annotation is incorrect.
  return 'string'
```

The return type of the functions `foo` and `bar` should be `str` because those
functions return a string, which is true whether the the functions are decorated
or not.

See https://www.python.org/dev/peps/pep-0318/ for more information about Python
decorators.

*   `getattr()`

Pytype does not know how to parse classes whose attributes are provides using
the [`getattr()`](https://docs.python.org/3/library/functions.html#getattr)
function. TensorFlow Federated makes uses of `getattr()` in classes such as
`tff.Struct`, `tff.Value`, and `tff.StructType` and these classes will not be
analyzed correctly by Pytype.

*   Pattern-Matching

Pytype does not handle pattern matching well before Python 3.10. TensorFlow
Federated makes heavy use of user-defined type guards (i.e. type guards other
than [`isinstance`](https://docs.python.org/3/library/functions.html#isinstance)
for performance reasons and Pytype can not interpret these type guards. This can
be fixed by either inserting `typing.cast` or locally disabling Pytype; however,
because of the use of user-defined type guards is so prevalent in parts of
TensorFlow Federated both of these fixes end up making the Python code harder to
read.

Note: Python 3.10 added supporting for
[User-Defined Type Guards](https://www.python.org/dev/peps/pep-0647/) so this
issue can be resolved after Python 3.10 is the minimum version of Python
TensorFlow Federated supports.

## Usage Of Pytype In TensorFlow Federated

TensorFlow Federated **does** use Python annotations and the Pytype analyzer.
However, it is *sometimes* helpful to not use Python annotations or to disable
Pytype. If diabling Pytype locally make the Python code harder to read, it is
preferrable to
[disable all pytype checks for a particular file](https://google.github.io/pytype/faq.html#how-do-i-disable-all-pytype-checks-for-a-particular-file).
