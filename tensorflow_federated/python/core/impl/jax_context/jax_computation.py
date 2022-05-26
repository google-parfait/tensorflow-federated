# Copyright 2020, The TensorFlow Federated Authors.
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
"""Definitions of experimental computation wrapper instances."""

from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.jax_context import jax_serialization


def _jax_strategy_fn(fn_to_wrap, fn_name, parameter_type, unpack):
  """Serializes a Python function containing JAX code as a TFF computation.

  Args:
    fn_to_wrap: The Python function containing JAX code to be serialized as a
      computation containing XLA.
    fn_name: The name for the constructed computation (currently ignored).
    parameter_type: An instance of `computation_types.Type` that represents the
      TFF type of the computation parameter, or `None` if there's none.
    unpack: See `unpack` in `function_utils.create_argument_unpacking_fn`.

  Returns:
    An instance of `computation_impl.ConcreteComputation` with the constructed
    computation.
  """
  del fn_name  # Unused.
  unpack_arguments_fn = function_utils.create_argument_unpacking_fn(
      fn_to_wrap, parameter_type, unpack=unpack)
  ctx_stack = context_stack_impl.context_stack
  comp_pb = jax_serialization.serialize_jax_computation(fn_to_wrap,
                                                        unpack_arguments_fn,
                                                        parameter_type,
                                                        ctx_stack)
  return computation_impl.ConcreteComputation(comp_pb, ctx_stack)


jax_computation = computation_wrapper.ComputationWrapper(_jax_strategy_fn)
jax_computation.__doc__ = (
    """Decorates/wraps Python functions containing JAX code as TFF computations.

  This wrapper can be used in a similar manner to `tff.tf_computation`, with
  exception of the following:

  * The code in the wrapped Python function must be JAX code that can be
    compiled to XLA (e.g., code that one would expect to be able to annotate
    with `@jax.jit`).

  * The inputs and outputs must be tensors, or (possibly recursively) nested
    structures of tensors. Sequences are currently not supported.

  Example:

  ```
  @tff.jax_computation(tf.int32)
  def comp(x):
    return jax.numpy.add(x, np.int32(10))
  ```
  """)
