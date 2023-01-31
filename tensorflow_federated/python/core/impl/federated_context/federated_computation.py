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
"""Definition of a federated computation."""

from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation_utils


def _federated_computation_wrapper_fn(parameter_type, name, **kwargs):
  """Wrapper function to plug orchestration logic into the TFF framework."""
  del kwargs  # Unused.
  ctx_stack = context_stack_impl.context_stack
  if parameter_type is None:
    parameter_name = None
  else:
    parameter_name = 'arg'
  fn_generator = federated_computation_utils.federated_computation_serializer(
      parameter_name=parameter_name,
      parameter_type=parameter_type,
      context_stack=ctx_stack,
      suggested_name=name,
  )
  arg = next(fn_generator)
  try:
    result = yield arg
  except Exception as e:  # pylint: disable=broad-except
    fn_generator.throw(e)
  target_lambda, extra_type_spec = fn_generator.send(result)
  fn_generator.close()
  yield computation_impl.ConcreteComputation(
      target_lambda.proto, ctx_stack, extra_type_spec
  )


federated_computation = computation_wrapper.ComputationWrapper(
    computation_wrapper.PythonTracingStrategy(_federated_computation_wrapper_fn)
)
federated_computation.__doc__ = """Decorates/wraps Python functions as TFF federated/composite computations.

  The term *federated computation* as used here refers to any computation that
  uses TFF programming abstractions. Examples of such computations may include
  federated training or federated evaluation that involve both client-side and
  server-side logic and involve network communication. However, this
  decorator/wrapper can also be used to construct composite computations that
  only involve local processing on a client or on a server.

  The main feature that distinguishes *federated computation* function bodies
  in Python from the bodies of TensorFlow defuns is that whereas in the latter,
  one slices and dices `tf.Tensor` instances using a variety of TensorFlow ops,
  in the former one slices and dices `tff.Value` instances using TFF operators.

  The supported modes of usage are identical to those for `tff.tf_computation`.

  Example:

    ```python
    @tff.federated_computation((tff.FunctionType(tf.int32, tf.int32), tf.int32))
    def foo(f, x):
      return f(f(x))
    ```

    The above defines `foo` as a function that takes a tuple consisting of an
    unary integer operator as the first element, and an integer as the second
    element, and returns the result of applying the unary operator to the
    integer twice. The body of `foo` does not contain federated communication
    operators, but we define it with `tff.federated_computation` as it can be
    used as building block in any section of TFF code (except inside sections
    of pure TensorFlow logic).

  Args:
    *args: Either a Python function, or TFF type spec, or both (function first),
      or neither. See also `tff.tf_computation` for an extended documentation.

  Returns:
    If invoked with a function as an argument, returns an instance of a TFF
    computation constructed based on this function. If called without one, as
    in the typical decorator style of usage, returns a callable that expects
    to be called with the function definition supplied as a parameter. See
    also `tff.tf_computation` for an extended documentation.
  """
