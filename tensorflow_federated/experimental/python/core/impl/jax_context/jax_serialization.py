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
"""Experimental utilities for serializing JAX computations."""

import jax
import numpy as np

from tensorflow_federated.experimental.python.core.impl.jax_context import jax_computation_context
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.backends.xla import xla_serialization
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import typed_object


class _XlaSerializerTensorArg(jax.ShapeDtypeStruct, typed_object.TypedObject):
  """Represents tensor type info understood by both TFF and JAX serializer."""

  def __init__(self, tensor_type, tensor_index):
    py_typecheck.check_type(tensor_type, computation_types.TensorType)
    shape = tuple(tensor_type.shape.as_list())
    dtype = tensor_type.dtype.as_numpy_dtype
    jax.ShapeDtypeStruct.__init__(self, shape, dtype)
    self._type_signature = tensor_type
    self._tensor_index = tensor_index

  @property
  def type_signature(self):
    return self._type_signature

  @property
  def tensor_index(self):
    return self._tensor_index


class _XlaSerializerStructArg(structure.Struct, typed_object.TypedObject):
  """Represents struct type info understood by both TFF and JAX serializer."""

  def __init__(self, type_spec, elements):
    py_typecheck.check_type(type_spec, computation_types.StructType)
    structure.Struct.__init__(self, elements)
    self._type_signature = type_spec

  @property
  def type_signature(self):
    return self._type_signature

  def __str__(self):
    return '_XlaSerializerStructArg({})'.format(structure.Struct.__str__(self))


def _tff_type_to_xla_serializer_arg(type_spec):
  """Converts TFF type into an argument for the JAX-to-XLA serializer.

  Args:
    type_spec: An instance of `computation_types.TensorType`.

  Returns:
    An object that carries both TFF and JAX type info, to be fed into the JAX
    serializer.
  """

  def _make(type_spec, next_unused_tensor_index):
    if isinstance(type_spec, computation_types.TensorType):
      obj = _XlaSerializerTensorArg(type_spec, next_unused_tensor_index)
      next_unused_tensor_index = next_unused_tensor_index + 1
      return obj, next_unused_tensor_index
    py_typecheck.check_type(type_spec, computation_types.StructType)
    elements = []
    for k, v in structure.to_elements(type_spec):
      obj, next_unused_tensor_index = _make(v, next_unused_tensor_index)
      elements.append((k, obj))
    obj = _XlaSerializerStructArg(type_spec, elements)
    return obj, next_unused_tensor_index

  obj, _ = _make(type_spec, 0)
  return obj


def _jax_shape_dtype_struct_to_tff_tensor(val):
  """Converts `jax.ShapeDtypeStruct` to `computation_types.TensorType`.

  Args:
    val: An instance of `jax.ShapeDtypeStruct`.

  Returns:
    A corresponding instance of `computation_types.TensorType`.

  Raises:
    TypeError: if arg type mismatches.
  """
  py_typecheck.check_type(val, jax.ShapeDtypeStruct)
  return computation_types.TensorType(val.dtype, val.shape)


def serialize_jax_computation(traced_fn, arg_fn, parameter_type, context_stack):
  """Serializes a Python function containing JAX code as a TFF computation.

  Args:
    traced_fn: The Python function containing JAX code to be traced by JAX and
      serialized as a TFF computation containing XLA code.
    arg_fn: An unpacking function that takes a TFF argument, and returns a combo
      of (args, kwargs) to invoke `traced_fn` with (e.g., as the one constructed
      by `function_utils.create_argument_unpacking_fn`).
    parameter_type: An instance of `computation_types.Type` that represents the
      TFF type of the computation parameter, or `None` if the function does not
      take any parameters.
    context_stack: The context stack to use during serialization.

  Returns:
    An instance of `pb.Computation` with the constructed computation.

  Raises:
    TypeError: if the arguments are of the wrong types.
  """
  py_typecheck.check_callable(traced_fn)
  py_typecheck.check_callable(arg_fn)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)

  if parameter_type is not None:
    parameter_type = computation_types.to_type(parameter_type)
    packed_arg = _tff_type_to_xla_serializer_arg(parameter_type)
  else:
    packed_arg = None

  args, kwargs = arg_fn(packed_arg)

  # While the fake parameters are fed via args/kwargs during serialization,
  # it is possible for them to get reorderd in the actual generate XLA code.
  # We use here the same flatenning function as that one, which is used by
  # the JAX serializer to determine the orderding and allow it to be captured
  # in the parameter binding. We do not need to do anything special for the
  # results, since the results, if multiple, are always returned as a tuple.
  flattened_obj, _ = jax.tree_util.tree_flatten((args, kwargs))
  tensor_indexes = list(np.argsort([x.tensor_index for x in flattened_obj]))

  def _adjust_arg(x):
    if isinstance(x, structure.Struct):
      return type_conversions.type_to_py_container(x, x.type_signature)
    else:
      return x

  args = [_adjust_arg(x) for x in args]
  kwargs = {k: _adjust_arg(v) for k, v in kwargs.items()}

  context = jax_computation_context.JaxComputationContext()
  with context_stack.install(context):
    tracer_callable = jax.xla_computation(
        traced_fn, tuple_args=True, return_shape=True)
    compiled_xla, returned_shape = tracer_callable(*args, **kwargs)

  if isinstance(returned_shape, jax.ShapeDtypeStruct):
    returned_type_spec = _jax_shape_dtype_struct_to_tff_tensor(returned_shape)
  else:
    returned_type_spec = computation_types.to_type(
        structure.map_structure(
            _jax_shape_dtype_struct_to_tff_tensor,
            structure.from_container(returned_shape, recursive=True)))

  computation_type = computation_types.FunctionType(parameter_type,
                                                    returned_type_spec)
  return xla_serialization.create_xla_tff_computation(compiled_xla,
                                                      tensor_indexes,
                                                      computation_type)
