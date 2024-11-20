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


from collections.abc import Callable, Sequence
from typing import Optional, Union

import federated_language
from federated_language.proto import computation_pb2 as pb
import jax
import numpy as np

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.jax_frontend import jax_computation_context
from tensorflow_federated.python.core.environments.xla_backend import xla_serialization


class _XlaSerializerTensorArg(
    jax.ShapeDtypeStruct, federated_language.TypedObject
):
  """Represents tensor type info understood by both TFF and JAX serializer."""

  def __init__(
      self, tensor_type: federated_language.TensorType, tensor_index: int
  ):
    py_typecheck.check_type(tensor_type, federated_language.TensorType)
    jax.ShapeDtypeStruct.__init__(self, tensor_type.shape, tensor_type.dtype)
    self._type_signature = tensor_type
    self._tensor_index = tensor_index

  @property
  def type_signature(self) -> federated_language.TensorType:
    return self._type_signature

  @property
  def tensor_index(self) -> int:
    return self._tensor_index


@jax.tree_util.register_pytree_node_class
class _XlaSerializerStructArg(structure.Struct, federated_language.TypedObject):
  """Represents struct type info understood by both TFF and JAX serializer."""

  def __init__(
      self,
      type_spec: federated_language.StructType,
      elements: Sequence[tuple[Optional[str], object]],
  ):
    py_typecheck.check_type(type_spec, federated_language.StructType)
    structure.Struct.__init__(self, elements)
    self._type_signature = type_spec

  @property
  def type_signature(self) -> federated_language.StructType:
    return self._type_signature

  def __str__(self) -> str:
    return f'_XlaSerializerStructArg({structure.Struct.__str__(self)})'

  def tree_flatten(
      self,
  ) -> tuple[
      tuple[Union[_XlaSerializerTensorArg, '_XlaSerializerStructArg'], ...],
      federated_language.StructType,
  ]:
    return tuple(self), self._type_signature

  @classmethod
  def tree_unflatten(
      cls,
      aux_data: federated_language.StructType,
      children: tuple[
          Union[_XlaSerializerTensorArg, '_XlaSerializerStructArg'], ...
      ],
  ) -> '_XlaSerializerStructArg':
    return cls(
        type_spec=aux_data,
        elements=tuple(zip(structure.name_list_with_nones(aux_data), children)),
    )


def _tff_type_to_xla_serializer_arg(
    type_spec: federated_language.Type,
) -> Union[_XlaSerializerStructArg, _XlaSerializerTensorArg]:
  """Converts TFF type into an argument for the JAX-to-XLA serializer.

  Args:
    type_spec: An instance of `federated_language.Type` containing only
      structure and tensor elements.

  Returns:
    An object that carries both TFF and JAX type info, to be fed into the JAX
    serializer.
  """

  def _undefined_shape_predicate(type_element: federated_language.Type) -> bool:
    if not isinstance(type_element, federated_language.TensorType):
      return False
    return not federated_language.array_shape_is_fully_defined(
        type_element.shape
    )

  has_undefined_shapes = federated_language.framework.type_contains(
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
      type_spec: federated_language.Type, next_unused_tensor_index: int
  ) -> tuple[Union[_XlaSerializerStructArg, _XlaSerializerTensorArg], int]:
    if isinstance(type_spec, federated_language.TensorType):
      obj = _XlaSerializerTensorArg(type_spec, next_unused_tensor_index)
      next_unused_tensor_index = next_unused_tensor_index + 1
      return obj, next_unused_tensor_index
    elif isinstance(type_spec, federated_language.StructType):
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
) -> federated_language.TensorType:
  """Converts `jax.ShapeDtypeStruct` to `federated_language.TensorType`.

  Args:
    val: An instance of `jax.ShapeDtypeStruct`.

  Returns:
    A corresponding instance of `federated_language.TensorType`.

  Raises:
    TypeError: if arg type mismatches.
  """
  return federated_language.TensorType(val.dtype, val.shape)


def serialize_jax_computation(
    fn: Callable[..., object],
    parameter_type: Union[
        federated_language.StructType, federated_language.TensorType
    ],
    context_stack: federated_language.framework.ContextStack,
) -> tuple[pb.Computation, federated_language.FunctionType]:
  """Serializes a Python function containing JAX code as a TFF computation.

  Args:
    fn: The Python function containing JAX code to be traced by JAX and
      serialized as a TFF computation containing XLA code.
    parameter_type: An instance of `federated_language.Type` that represents the
      TFF type of the computation parameter, or `None` if the function does not
      take any parameters.
    context_stack: The context stack to use during serialization.

  Returns:
    A 2-tuple of `pb.Computation` with the constructed computation and a
    `federated_language.FunctionType` containing the full type including
    Python container annotations.

  Raises:
    TypeError: if the arguments are of the wrong types.
  """
  py_typecheck.check_type(
      context_stack, federated_language.framework.ContextStack
  )

  if parameter_type is not None:
    parameter_type = federated_language.to_type(parameter_type)
    packed_arg = _tff_type_to_xla_serializer_arg(parameter_type)
  else:
    packed_arg = None

  args, kwargs = federated_language.framework.unpack_arg(
      fn, parameter_type, packed_arg
  )

  # While the fake parameters are fed via args/kwargs during serialization,
  # it is possible for them to get reordered in the actual generated XLA code.
  # We use here the same flattening function as that one, which is used by
  # the JAX serializer to determine the ordering and allow it to be captured
  # in the parameter binding. We do not need to do anything special for the
  # results, since the results, if multiple, are always returned as a tuple.
  flattened_obj, _ = jax.tree_util.tree_flatten((args, kwargs))
  tensor_indexes = list(np.argsort([x.tensor_index for x in flattened_obj]))

  context = jax_computation_context.JaxComputationContext()
  # TODO: b/347811116 - Remove this version check when the JAX version can be
  # upgraded.
  if jax.__version_info__ > (0, 4, 29):
    with context_stack.install(context):
      lowered = jax.jit(fn, keep_unused=True).lower(*args, **kwargs)
      compiled_xla = lowered.compiler_ir('hlo')

    # Test if the output is a tuple, or a single array and construct the
    # return spec accordingly.
    if isinstance(lowered.out_info, jax.stages.OutInfo):
      returned_type_spec = _jax_shape_dtype_struct_to_tff_tensor(
          jax.ShapeDtypeStruct(
              shape=lowered.out_info.shape, dtype=lowered.out_info.dtype
          )
      )
    else:
      returned_type_spec = federated_language.to_type(
          jax.tree_util.tree_map(
              _jax_shape_dtype_struct_to_tff_tensor, lowered.out_info
          )
      )
  else:
    with context_stack.install(context):
      tracer_callable = jax.xla_computation(fn, return_shape=True)  # type: ignore
      compiled_xla, returned_shape = tracer_callable(*args, **kwargs)

    if isinstance(returned_shape, jax.ShapeDtypeStruct):
      returned_type_spec = _jax_shape_dtype_struct_to_tff_tensor(returned_shape)
    else:
      returned_type_spec = federated_language.to_type(
          jax.tree_util.tree_map(
              _jax_shape_dtype_struct_to_tff_tensor, returned_shape
          )
      )

  computation_type = federated_language.FunctionType(
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
) -> tuple[tuple[object, ...], tuple[Optional[str], ...]]:
  child_names, child_values = tuple(zip(*structure.iter_elements(struct)))
  return (child_values, child_names)


def _struct_unflatten(
    child_names: tuple[Optional[str], ...], child_values: tuple[object, ...]
) -> structure.Struct:
  return structure.Struct(tuple(zip(child_names, child_values)))


jax.tree_util.register_pytree_node(
    structure.Struct, _struct_flatten, _struct_unflatten
)
