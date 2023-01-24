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
"""Utilities for serializing JAX computations."""


from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, Union

import jax
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.jax_context import jax_computation_context
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import typed_object
from tensorflow_federated.python.core.impl.xla_context import xla_serialization


class _XlaSerializerTensorArg(jax.ShapeDtypeStruct, typed_object.TypedObject):
  """Represents tensor type info understood by both TFF and JAX serializer."""

  def __init__(
      self, tensor_type: computation_types.TensorType, tensor_index: int
  ):
    py_typecheck.check_type(tensor_type, computation_types.TensorType)
    # We assume shape has already been checked to be fully defined here.
    shape = tuple(tensor_type.shape.as_list())
    dtype = tensor_type.dtype.as_numpy_dtype
    jax.ShapeDtypeStruct.__init__(self, shape, dtype)
    self._type_signature = tensor_type
    self._tensor_index = tensor_index

  @property
  def type_signature(self) -> computation_types.TensorType:
    return self._type_signature

  @property
  def tensor_index(self) -> int:
    return self._tensor_index


@jax.tree_util.register_pytree_node_class
class _XlaSerializerStructArg(structure.Struct, typed_object.TypedObject):
  """Represents struct type info understood by both TFF and JAX serializer."""

  def __init__(
      self, type_spec: computation_types.StructType, elements: Sequence[Any]
  ):
    py_typecheck.check_type(type_spec, computation_types.StructType)
    structure.Struct.__init__(self, elements)
    self._type_signature = type_spec

  @property
  def type_signature(self) -> computation_types.StructType:
    return self._type_signature

  def __str__(self) -> str:
    return f'_XlaSerializerStructArg({structure.Struct.__str__(self)})'

  def tree_flatten(
      self,
  ) -> tuple[
      tuple[Union[_XlaSerializerTensorArg, '_XlaSerializerStructArg'], ...],
      computation_types.StructType,
  ]:
    return tuple(self), self._type_signature

  @classmethod
  def tree_unflatten(
      cls,
      aux_data: computation_types.StructType,
      children: tuple[
          Union[_XlaSerializerTensorArg, '_XlaSerializerStructArg'], ...
      ],
  ) -> '_XlaSerializerStructArg':
    return cls(
        type_spec=aux_data,
        elements=tuple(zip(structure.name_list_with_nones(aux_data), children)),
    )


def _tff_type_to_xla_serializer_arg(
    type_spec: computation_types.Type,
) -> Union[_XlaSerializerStructArg, _XlaSerializerTensorArg]:
  """Converts TFF type into an argument for the JAX-to-XLA serializer.

  Args:
    type_spec: An instance of `computation_types.Type` containing only structure
      and tensor elements.

  Returns:
    An object that carries both TFF and JAX type info, to be fed into the JAX
    serializer.
  """

  def _undefined_shape_predicate(type_element: computation_types.Type) -> bool:
    if type_element.is_tensor():
      if not type_element.shape.is_fully_defined():
        return True
    return False

  has_undefined_shapes = type_analysis.contains(
      type_spec, _undefined_shape_predicate
  )
  if has_undefined_shapes:
    raise TypeError(
        'Can only serialize XLA computations whose parameters '
        'contain fully-defined TensorShapes at the leaves; '
        'encountered undefined tensor shapes (or unknown rank '
        'tensors) in the signature:\n'
        f'{type_spec.formatted_representation()}'
    )

  def _make(
      type_spec: computation_types.Type, next_unused_tensor_index: int
  ) -> tuple[Union[_XlaSerializerStructArg, _XlaSerializerTensorArg], int]:
    if type_spec.is_tensor():
      obj = _XlaSerializerTensorArg(type_spec, next_unused_tensor_index)
      next_unused_tensor_index = next_unused_tensor_index + 1
      return obj, next_unused_tensor_index
    elif type_spec.is_struct():
      elements = []
      for k, v in structure.to_elements(type_spec):
        obj, next_unused_tensor_index = _make(v, next_unused_tensor_index)
        elements.append((k, obj))
      obj = _XlaSerializerStructArg(type_spec, elements)
      return obj, next_unused_tensor_index
    else:
      raise TypeError(
          'Can only construct an XLA serializer for TFF types '
          'which contain exclusively structure and tensor types; '
          f'found type:\n {type_spec.formatted_representation()}'
      )

  obj, _ = _make(type_spec, 0)
  return obj


def _jax_shape_dtype_struct_to_tff_tensor(
    val: jax.ShapeDtypeStruct,
) -> computation_types.TensorType:
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


def serialize_jax_computation(
    traced_fn: Callable[..., Any],
    arg_fn: Callable[
        [Union[_XlaSerializerStructArg, _XlaSerializerTensorArg]],
        tuple[Sequence[Any], Mapping[str, Any]],
    ],
    parameter_type: Union[
        computation_types.StructType, computation_types.TensorType
    ],
    context_stack: context_stack_base.ContextStack,
) -> tuple[pb.Computation, computation_types.FunctionType]:
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
    A 2-tuple of `pb.Computation` with the constructed computation and a
    `computation_types.FunctionType` containing the full type including
    Python container annotations.

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
  # it is possible for them to get reordered in the actual generated XLA code.
  # We use here the same flattening function as that one, which is used by
  # the JAX serializer to determine the ordering and allow it to be captured
  # in the parameter binding. We do not need to do anything special for the
  # results, since the results, if multiple, are always returned as a tuple.
  flattened_obj, _ = jax.tree_util.tree_flatten((args, kwargs))
  tensor_indexes = list(np.argsort([x.tensor_index for x in flattened_obj]))

  context = jax_computation_context.JaxComputationContext()
  with context_stack.install(context):
    tracer_callable = jax.xla_computation(
        traced_fn, tuple_args=False, return_shape=True
    )
    compiled_xla, returned_shape = tracer_callable(*args, **kwargs)

  if isinstance(returned_shape, jax.ShapeDtypeStruct):
    returned_type_spec = _jax_shape_dtype_struct_to_tff_tensor(returned_shape)
  else:
    returned_type_spec = computation_types.to_type(
        jax.tree_util.tree_map(
            _jax_shape_dtype_struct_to_tff_tensor, returned_shape
        )
    )

  computation_type = computation_types.FunctionType(
      parameter_type, returned_type_spec
  )
  return (
      xla_serialization.create_xla_tff_computation(
          compiled_xla, tensor_indexes, computation_type
      ),
      computation_type,
  )


# Registers TFF's Struct as a node that Jax's tree-traversal utilities can walk
# through. Pytree flattening works _per-level_ of the tree, so we don't
# use structure.flatten and structure.unflatten here, rather we only unpack
# the immediate Struct, and let pytrees apply flattening recursively to properly
# pack/unpack intermediate contaienrs of other types.


def _struct_flatten(
    struct: structure.Struct,
) -> tuple[tuple[Any, ...], tuple[Optional[str], ...]]:
  child_names, child_values = tuple(zip(*structure.iter_elements(struct)))
  return (child_values, child_names)


def _struct_unflatten(
    child_names: tuple[Optional[str], ...], child_values: tuple[Any, ...]
) -> structure.Struct:
  return structure.Struct(tuple(zip(child_names, child_values)))


jax.tree_util.register_pytree_node(
    structure.Struct, _struct_flatten, _struct_unflatten
)
