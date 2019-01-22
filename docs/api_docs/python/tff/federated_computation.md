<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_computation" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_computation

``` python
tff.federated_computation(*args)
```



Defined in [`core/api/computations.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computations.py).

Decorates/wraps Python functions as TFF federated/composite computations.

The term *federated computation* as used here refers to any computation that
uses TFF programming abstractions. Examples of such computations may include
federated training or federated evaluation that involve both client-side and
server-side logic and involve network communication. However, this
decorator/wrapper can also be used to construct composite computations that
only involve local processing on a client or on a server.

The main feature that distinguishes *federated computation* function bodies
in Python from the bodies of TensorFlow defuns is that whereas in the latter,
one slices and dices tf.Tensor instances using a variety of TensorFlow ops,
in the former one slices and dices tff.Value instances using TFF operators.

The supported modes of usage are identical to those for `tf_computation`.

Example:

  ```
  @federated_computation((types.FunctionType(tf.int32, tf.int32), tf.int32))
  def foo(f, x):
    return f(f(x))
  ```

  The above defines `foo` as a function that takes a tuple consisting of an
  unary integer operator as the first element, and an integer as the second
  element, and returns the result of applying the unary operator to the
  integer twice. The body of 'foo' does not contain federated communication
  operators, but we define it with '@federated_computation' as it can be
  used as building block in any section of TFF code (except inside sections
  of pure TensorFlow logic).

#### Args:

* <b>`*args`</b>: Either a Python function, or TFF type spec, or both (function first),
    or neither. See also `tf_computation` for an extended documentation.


#### Returns:

If invoked with a function as an argument, returns an instance of a TFF
computation constructed based on this function. If called without one, as
in the typical decorator style of usage, returns a callable that expects
to be called with the function definition supplied as a parameter. See
also `tf_computation` for an extended documentation.