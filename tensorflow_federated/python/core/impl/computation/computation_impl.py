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
"""Defines the implementation of the base Computation interface."""

from collections.abc import Callable
from typing import Optional

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization


class ConcreteComputation(computation_base.Computation):
  """A representation of a `pb.Computation` in the `tff.Computation` interface.

  This implementation exposes methods to retrieve the backing `pb.Computation`,
  as well as the Python representation of this protocol buffer represented by
  an instance of `building_blocks.ComputationBuildingBlock`.
  """

  @classmethod
  def get_proto(cls, value: 'ConcreteComputation') -> pb.Computation:
    py_typecheck.check_type(value, cls)
    return value._computation_proto  # pylint: disable=protected-access

  @classmethod
  def with_type(
      cls,
      value: 'ConcreteComputation',
      type_spec: computation_types.FunctionType,
  ) -> 'ConcreteComputation':
    py_typecheck.check_type(value, cls)
    py_typecheck.check_type(type_spec, computation_types.Type)
    # Ensure we are assigning a type-safe signature.
    value.type_signature.check_assignable_from(type_spec)
    # pylint: disable=protected-access
    return cls(
        value._computation_proto, value._context_stack, annotated_type=type_spec
    )
    # pylint: enable=protected-access

  @classmethod
  def from_building_block(
      cls, building_block: building_blocks.ComputationBuildingBlock
  ) -> 'ConcreteComputation':
    """Converts a computation building block to a computation impl."""
    py_typecheck.check_type(
        building_block, building_blocks.ComputationBuildingBlock
    )
    return cls(
        building_block.proto,
        context_stack_impl.context_stack,
        annotated_type=building_block.type_signature,  # pytype: disable=wrong-arg-types
    )

  def to_building_block(self):
    # TODO: b/161560999 - currently destroys annotated type.
    # This should perhaps be fixed by adding `type_parameter` to `from_proto`.
    return building_blocks.ComputationBuildingBlock.from_proto(
        self._computation_proto
    )

  def to_compiled_building_block(self):
    return building_blocks.CompiledComputation(
        self._computation_proto, type_signature=self.type_signature
    )

  def __init__(
      self,
      computation_proto: pb.Computation,
      context_stack: context_stack_base.ContextStack,
      annotated_type: Optional[computation_types.FunctionType] = None,
      transform_result: Optional[Callable[[object], object]] = None,
  ):
    """Constructs a new instance of ConcreteComputation from the computation_proto.

    Args:
      computation_proto: The protocol buffer that represents the computation, an
        instance of pb.Computation.
      context_stack: The context stack to use.
      annotated_type: Optional, type information with additional annotations
        that replaces the information in `computation_proto.type`.
      transform_result: An `Optional` `Callable` used to transform the result
        before it is returned.

    Raises:
      TypeError: If `annotated_type` is not `None` and is not compatible with
        `computation_proto.type`.
      ValueError: If `computation_proto.type` is `None`.
    """
    py_typecheck.check_type(computation_proto, pb.Computation)
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    if computation_proto.type is None:
      raise ValueError('Expected `computation_proto.type` to not be `None`.')
    type_spec = type_serialization.deserialize_type(computation_proto.type)

    if annotated_type is not None:
      if type_spec is None or not type_spec.is_assignable_from(annotated_type):
        raise TypeError(
            'annotated_type not compatible with computation_proto.type\n'
            f'computation_proto.type: {type_spec}\n'
            f'annotated_type: {annotated_type}'
        )
      type_spec = annotated_type

    if not isinstance(type_spec, computation_types.FunctionType):
      raise TypeError(
          f'{type_spec} is not a functional type, from proto: '
          f'{computation_proto}'
      )

    self._type_signature = type_spec
    self._context_stack = context_stack
    self._computation_proto = computation_proto
    self._transform_result = transform_result

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, ConcreteComputation):
      return NotImplemented
    return self._computation_proto == other._computation_proto

  @property
  def type_signature(self) -> computation_types.FunctionType:
    return self._type_signature

  def __call__(self, *args, **kwargs):
    arg = function_utils.pack_args(
        self._type_signature.parameter,  # pytype: disable=attribute-error
        args,
        kwargs,
    )
    result = self._context_stack.current.invoke(self, arg)
    if self._transform_result is not None:
      result = self._transform_result(result)
    return result

  def __hash__(self) -> int:
    return hash(self._computation_proto.SerializeToString(deterministic=True))
