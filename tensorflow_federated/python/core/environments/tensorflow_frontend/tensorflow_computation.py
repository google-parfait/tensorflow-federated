# Copyright 2018, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Definition of a tensorflow computation."""

from typing import Optional

import tensorflow as tf
import tree

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_serialization
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions


def _to_numpy(value: object) -> object:
  """Convert `value` to a numpy value."""

  def _fn(obj):
    if isinstance(obj, tf.Variable):
      return obj.read_value().numpy()
    elif isinstance(obj, tf.Tensor) and not tf.is_symbolic_tensor(obj):
      return obj.numpy()
    else:
      return None

  # Important: `tree.traverse` is used instead of `tree.map_structure`, even
  # though mutating the structure of `value` is not required, because the
  # `tree.map_structure` sorts `dict` keys.
  return tree.traverse(_fn, value)


def transform_args(args: object) -> object:
  """Transform the arguments to TensorFlow computations."""
  return _to_numpy(args)


def transform_result(result: object) -> object:
  """Transforms the result of TensorFlow computations."""
  return _to_numpy(result)


def _tf_wrapper_fn(
    fn,
    parameter_type,
    unpack: Optional[bool],
    name: Optional[str] = None,
):
  """Wrapper function to plug Tensorflow logic into the TFF framework."""
  del name  # Unused.
  if not type_analysis.is_tensorflow_compatible_type(parameter_type):
    raise TypeError(
        '`tff.tensorflow.computation`s can accept only parameter types with '
        'constituents `SequenceType`, `StructType` '
        'and `TensorType`; you have attempted to create one '
        'with the type {}.'.format(parameter_type)
    )

  fn = function_utils.wrap_as_zero_or_one_arg_callable(
      fn, parameter_type, unpack
  )
  context_stack = context_stack_impl.context_stack
  comp_pb, extra_type_spec = (
      tensorflow_serialization.serialize_py_fn_as_tf_computation(
          fn, parameter_type, context_stack
      )
  )
  return computation_impl.ConcreteComputation(
      computation_proto=comp_pb,
      context_stack=context_stack,
      annotated_type=extra_type_spec,
  )


tf_computation = computation_wrapper.ComputationWrapper(
    _tf_wrapper_fn,
    computation_types.tensorflow_to_type,
    type_conversions.tensorflow_infer_type,
)
tf_computation.__doc__ = """Decorates/wraps Python functions and defuns as TFF TensorFlow computations.

  This symbol can be used as either a decorator or a wrapper applied to a
  function given to it as an argument. The supported patterns and examples of
  usage are as follows:

  1. Convert an existing function inline into a TFF computation. This is the
     simplest mode of usage, and how one can embed existing non-TFF code for
     use with the TFF framework. In this mode, one invokes
     `tff.tensorflow.computation` with a pair of arguments, the first being a
     function/defun that contains the logic, and the second being the TFF type
     of the parameter:

     ```python
     foo = tff.tensorflow.computation(lambda x: x > 10, tf.int32)
     ```

     After executing the above code snippet, `foo` becomes an instance of the
     abstract base class `Computation`. Like all computations, it has the
     `type_signature` property:

     ```python
     str(foo.type_signature) == '(int32 -> bool)'
     ```

     The function passed as a parameter doesn't have to be a lambda, it can
     also be an existing Python function or a defun. Here's how to construct
     a computation from the standard TensorFlow operator `tf.add`:

     ```python
     foo = tff.tensorflow.computation(tf.add, (tf.int32, tf.int32))
     ```

     The resulting type signature is as expected:

     ```python
     str(foo.type_signature) == '(<int32,int32> -> int32)'
     ```

     If one intends to create a computation that doesn't accept any arguments,
     the type argument is simply omitted. The function must be a no-argument
     function as well:

     ```python
     foo = tff.tensorflow.computation(lambda: tf.constant(10))
     ```

  2. Decorate a Python function or a TensorFlow defun with a TFF type to wrap
     it as a TFF computation. The only difference between this mode of usage
     and the one mentioned above is that instead of passing the function/defun
     as an argument, `tff.tensorflow.computation` along with the optional type specifier
     is written above the function/defun's body.

     Here's an example of a computation that accepts a parameter:

     ```python
     @tff.tensorflow.computation(tf.int32)
     def foo(x):
       return x > 10
     ```

     One can think of this mode of usage as merely a syntactic sugar for the
     example already given earlier:

     ```python
     foo = tff.tensorflow.computation(lambda x: x > 10, tf.int32)
     ```

     Here's an example of a no-parameter computation:

     ```python
     @tff.tensorflow.computation
     def foo():
       return tf.constant(10)
     ```

     Again, this is merely syntactic sugar for the example given earlier:

     ```python
     foo = tff.tensorflow.computation(lambda: tf.constant(10))
     ```

     If the Python function has multiple decorators,
     `tff.tensorflow.computation` should be the outermost one (the one that
     appears first in the sequence).

  3. Create a polymorphic callable to be instantiated based on arguments,
     similarly to TensorFlow defuns that have been defined without an input
     signature.

     This mode of usage is symmetric to those above. One simply omits the type
     specifier, and applies `tff.tensorflow.computation` as a decorator or
     wrapper to a function/defun that does expect parameters.

     Here's an example of wrapping a lambda as a polymorphic callable:

     ```python
     foo = tff.tensorflow.computation(lambda x, y: x > y)
     ```

     The resulting `foo` can be used in the same ways as if it were had the
     type been declared; the corresponding computation is simply created on
     demand, in the same way as how polymorphic TensorFlow defuns create and
     cache concrete function definitions for each combination of argument
     types.

     ```python
     ...foo(1, 2)...
     ...foo(0.5, 0.3)...
     ```

     Here's an example of creating a polymorphic callable via decorator:

     ```python
     @tff.tensorflow.computation
     def foo(x, y):
       return x > y
     ```

     The syntax is symmetric to all examples already shown.

  Args:
    *args: Either a function/defun, or TFF type spec, or both (function first),
      or neither, as documented in the 3 patterns and examples of usage above.

  Returns:
    If invoked with a function as an argument, returns an instance of a TFF
    computation constructed based on this function. If called without one, as
    in the typical decorator style of usage, returns a callable that expects
    to be called with the function definition supplied as a parameter; see the
    patterns and examples of usage above.
  """
