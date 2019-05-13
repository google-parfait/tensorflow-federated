<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_computation" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_computation

Decorates/wraps Python functions as TFF federated/composite computations.

```python
tff.federated_computation(*args)
```

Defined in
[`core/api/computations.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computations.py).

<!-- Placeholder for "Used in" -->

The term *federated computation* as used here refers to any computation that
uses TFF programming abstractions. Examples of such computations may include
federated training or federated evaluation that involve both client-side and
server-side logic and involve network communication. However, this
decorator/wrapper can also be used to construct composite computations that only
involve local processing on a client or on a server.

The main feature that distinguishes *federated computation* function bodies in
Python from the bodies of TensorFlow defuns is that whereas in the latter, one
slices and dices `tf.Tensor` instances using a variety of TensorFlow ops, in the
former one slices and dices <a href="../tff/Value.md"><code>tff.Value</code></a>
instances using TFF operators.

The supported modes of usage are identical to those for
<a href="../tff/tf_computation.md"><code>tff.tf_computation</code></a>.

#### Example:

```python
@tff.federated_computation((tff.FunctionType(tf.int32, tf.int32), tf.int32))
def foo(f, x):
  return f(f(x))
```

The above defines `foo` as a function that takes a tuple consisting of an unary
integer operator as the first element, and an integer as the second element, and
returns the result of applying the unary operator to the integer twice. The body
of `foo` does not contain federated communication operators, but we define it
with
<a href="../tff/federated_computation.md"><code>tff.federated_computation</code></a>
as it can be used as building block in any section of TFF code (except inside
sections of pure TensorFlow logic).

#### Args:

*   <b>`*args`</b>: Either a Python function, or TFF type spec, or both
    (function first), or neither. See also
    <a href="../tff/tf_computation.md"><code>tff.tf_computation</code></a> for
    an extended documentation.

#### Returns:

If invoked with a function as an argument, returns an instance of a TFF
computation constructed based on this function. If called without one, as in the
typical decorator style of usage, returns a callable that expects to be called
with the function definition supplied as a parameter. See also
<a href="../tff/tf_computation.md"><code>tff.tf_computation</code></a> for an
extended documentation.
