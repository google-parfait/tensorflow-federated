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
"""Definitions of JAX computation wrapper instances."""

from collections.abc import Callable, Sequence
from typing import Optional, Union

import jax
import numpy as np

from tensorflow_federated.python.core.environments.jax_frontend import jax_serialization
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis


def _contains_dtype(
    type_spec: computation_types.Type,
    dtype: Union[type[np.generic], Sequence[type[np.generic]]],
) -> bool:
  """Returns `True` if `type_spec` contains the `dtype`."""
  if not isinstance(dtype, Sequence):
    dtype = [dtype]

  def predicate(type_spec: computation_types.Type) -> bool:
    return (
        isinstance(type_spec, computation_types.TensorType)
        and type_spec.dtype.type in dtype
    )

  return type_analysis.contains(type_spec, predicate)


def _jax_wrapper_fn(
    fn: Callable[..., object],
    parameter_type: Optional[
        Union[computation_types.StructType, computation_types.TensorType]
    ],
    unpack: Optional[bool],
    name: Optional[str] = None,
    **kwargs,
) -> computation_impl.ConcreteComputation:
  """Serializes a Python function containing JAX code as a TFF computation.

  Args:
    fn: The Python function containing JAX code to be serialized as a
      computation containing XLA.
    parameter_type: An instance of `computation_types.Type` that represents the
      TFF type of the computation parameter, or `None` if there's none.
    unpack: See `unpack` in `function_utils.wrap_as_zero_or_one_arg_callable`.
    name: The name for the constructed computation (currently ignored).
    **kwargs: Unused currently. A placeholder for passing Jax strategy specific
      parameters.

  Returns:
    An instance of `computation_impl.ConcreteComputation` with the constructed
    computation.
  """
  del unpack, name, kwargs  # Unused.

  if parameter_type is not None:
    if _contains_dtype(parameter_type, np.str_):
      raise ValueError(
          f'JAX does not support `np.str_` dtypes, found: {parameter_type}.'
      )

    if not jax.config.read('jax_enable_x64') and _contains_dtype(
        parameter_type, [np.int64, np.uint64, np.float64, np.complex128]
    ):
      raise ValueError(
          'Using x64-precision numbers in JAX is not enabled and found an x64'
          ' dtype. To use x64-precision numbers in JAX, you must set the'
          ' `jax_enable_x64` configuration variable. See'
          ' https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precisionJAX'
          f' for more information.\nFound: {parameter_type}'
      )

  context_stack = context_stack_impl.context_stack
  comp_pb, extra_type_spec = jax_serialization.serialize_jax_computation(
      fn, parameter_type, context_stack
  )
  return computation_impl.ConcreteComputation(
      computation_proto=comp_pb,
      context_stack=context_stack,
      annotated_type=extra_type_spec,
  )


jax_computation = computation_wrapper.ComputationWrapper(_jax_wrapper_fn)
jax_computation.__doc__ = """Decorates/wraps Python functions containing JAX code as TFF computations.

  This wrapper can be used in a similar manner to `tff.tensorflow.computation`,
  with exception of the following:

  * The code in the wrapped Python function must be JAX code that can be
    compiled to XLA (e.g., code that one would expect to be able to annotate
    with `@jax.jit`).

  * The inputs and outputs must be tensors, or (possibly recursively) nested
    structures of tensors. Sequences are currently not supported.

  Example:

  ```
  @tff.jax.computation(np.int32)
  def comp(x):
    return jax.numpy.add(x, np.int32(10))
  ```
  """
