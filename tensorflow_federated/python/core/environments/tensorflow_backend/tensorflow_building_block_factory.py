# Copyright 2019, The TensorFlow Federated Authors.
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
"""A library of construction functions for building block structures."""

from collections.abc import Callable
import functools
from typing import Optional, Union

import federated_language

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_computation_factory
from tensorflow_federated.python.core.environments.tensorflow_backend import type_conversions


@functools.lru_cache()
def _create_tensorflow_constant(
    type_spec: federated_language.Type,
    scalar_value: Union[int, float, str],
    name=None,
) -> federated_language.framework.Call:
  """Creates called graph returning constant `scalar_value` of type `type_spec`.

  `scalar_value` must be a scalar, and cannot be a float if any of the tensor
  leaves of `type_spec` contain an integer data type. `type_spec` must contain
  only named tuples and tensor types, but these can be arbitrarily nested.

  Args:
    type_spec: A `federated_language.Type` whose resulting type tree can only
      contain named tuples and tensors.
    scalar_value: Scalar value to place in all the tensor leaves of `type_spec`.
    name: An optional string name to use as the name of the computation.

  Returns:
    An instance of `federated_language.framework.Call`, whose argument is `None`
    and whose function is a noarg
    `federated_language.framework.CompiledComputation` which returns the
    specified `scalar_value` packed into a TFF structure of type `type_spec.

  Raises:
    TypeError: If the type assumptions above are violated.
  """
  proto, function_type = tensorflow_computation_factory.create_constant(
      scalar_value, type_spec
  )
  compiled = federated_language.framework.CompiledComputation(
      proto, name, type_signature=function_type
  )
  return federated_language.framework.Call(compiled, None)


def create_null_federated_aggregate() -> federated_language.framework.Call:
  """Creates an aggregate over an empty struct and returns an empty struct."""
  unit = federated_language.framework.Struct([])
  unit_type = unit.type_signature
  value = federated_language.framework.create_federated_value(
      unit, federated_language.CLIENTS
  )
  zero = unit
  accumulate_proto, accumulate_type = (
      tensorflow_computation_factory.create_binary_operator(
          lambda a, b: a, unit_type
      )
  )
  accumulate = federated_language.framework.CompiledComputation(
      accumulate_proto, type_signature=accumulate_type
  )
  merge = accumulate
  report_proto, report_type = tensorflow_computation_factory.create_identity(
      federated_language.StructType([])
  )
  report = federated_language.framework.CompiledComputation(
      report_proto, type_signature=report_type
  )
  return federated_language.framework.create_federated_aggregate(
      value, zero, accumulate, merge, report
  )


def create_null_federated_broadcast():
  return federated_language.framework.create_federated_broadcast(
      federated_language.framework.create_federated_value(
          federated_language.framework.Struct([]), federated_language.SERVER
      )
  )


def create_null_federated_map() -> federated_language.framework.Call:
  fn_proto, fn_type = tensorflow_computation_factory.create_identity(
      federated_language.StructType([])
  )
  fn = federated_language.framework.CompiledComputation(
      fn_proto, type_signature=fn_type
  )
  return federated_language.framework.create_federated_map(
      fn,
      federated_language.framework.create_federated_value(
          federated_language.framework.Struct([]), federated_language.CLIENTS
      ),
  )


def create_null_federated_secure_sum():
  return federated_language.framework.create_federated_secure_sum(
      federated_language.framework.create_federated_value(
          federated_language.framework.Struct([]), federated_language.CLIENTS
      ),
      federated_language.framework.Struct([]),
  )


def create_null_federated_secure_sum_bitwidth():
  return federated_language.framework.create_federated_secure_sum_bitwidth(
      federated_language.framework.create_federated_value(
          federated_language.framework.Struct([]), federated_language.CLIENTS
      ),
      federated_language.framework.Struct([]),
  )


@functools.lru_cache()
def create_generic_constant(
    type_spec: Optional[federated_language.Type],
    scalar_value: Union[int, float],
) -> federated_language.framework.ComputationBuildingBlock:
  """Creates constant for a combination of federated, tuple and tensor types.

  Args:
    type_spec: A `federated_language.Type` containing only federated, tuple or
      tensor types, or `None` to use to construct a generic constant.
    scalar_value: The scalar value we wish this constant to have.

  Returns:
    Instance of `federated_language.framework.ComputationBuildingBlock`
    representing `scalar_value` packed into `type_spec`.

  Raises:
    TypeError: If types don't match their specification in the args section.
      Notice validation of consistency of `type_spec` with `scalar_value` is not
      the rsponsibility of this function.
  """
  if type_spec is None:
    return _create_tensorflow_constant(type_spec, scalar_value)
  py_typecheck.check_type(type_spec, federated_language.Type)
  inferred_scalar_value_type = type_conversions.tensorflow_infer_type(
      scalar_value
  )
  if not isinstance(
      inferred_scalar_value_type, federated_language.TensorType
  ) or not federated_language.array_shape_is_scalar(
      inferred_scalar_value_type.shape
  ):
    raise TypeError(
        'Must pass a scalar value to `create_generic_constant`; encountered a '
        'value {}'.format(scalar_value)
    )

  def _check_parameters(type_spec: federated_language.Type) -> bool:
    return isinstance(
        type_spec,
        (
            federated_language.FederatedType,
            federated_language.StructType,
            federated_language.TensorType,
        ),
    )

  if not federated_language.framework.type_contains_only(
      type_spec, _check_parameters
  ):
    raise TypeError

  def _predicate(type_spec: federated_language.Type) -> bool:
    return isinstance(
        type_spec,
        (
            federated_language.StructType,
            federated_language.TensorType,
        ),
    )

  if federated_language.framework.type_contains_only(type_spec, _predicate):
    return _create_tensorflow_constant(type_spec, scalar_value)
  elif isinstance(type_spec, federated_language.FederatedType):
    unplaced_zero = _create_tensorflow_constant(type_spec.member, scalar_value)
    if type_spec.placement is federated_language.CLIENTS:
      placement_federated_type = federated_language.FederatedType(
          type_spec.member, type_spec.placement, all_equal=True
      )
      placement_fn_type = federated_language.FunctionType(
          type_spec.member, placement_federated_type
      )
      placement_function = federated_language.framework.Intrinsic(
          federated_language.framework.FEDERATED_VALUE_AT_CLIENTS.uri,
          placement_fn_type,
      )
    elif type_spec.placement is federated_language.SERVER:
      placement_federated_type = federated_language.FederatedType(
          type_spec.member, type_spec.placement, all_equal=True
      )
      placement_fn_type = federated_language.FunctionType(
          type_spec.member, placement_federated_type
      )
      placement_function = federated_language.framework.Intrinsic(
          federated_language.framework.FEDERATED_VALUE_AT_SERVER.uri,
          placement_fn_type,
      )
    else:
      raise NotImplementedError(
          f'Unexpected placement found: {type_spec.placement}.'
      )
    return federated_language.framework.Call(placement_function, unplaced_zero)
  elif isinstance(type_spec, federated_language.StructType):
    elements = []
    for i, _ in enumerate(type_spec):
      elements.append(create_generic_constant(type_spec[i], scalar_value))
    names = [name for name, _ in structure.iter_elements(type_spec)]
    packed_elements = federated_language.framework.Struct(elements)
    named_tuple = federated_language.framework.create_named_tuple(
        packed_elements,
        names,
        type_spec.python_container,
    )
    return named_tuple
  else:
    raise ValueError(
        'The type_spec {} has slipped through all our '
        'generic constant cases, and failed to raise.'.format(type_spec)
    )


def apply_binary_operator_with_upcast(
    arg: federated_language.framework.ComputationBuildingBlock,
    operator: Callable[[object, object], object],
) -> federated_language.framework.Call:
  """Constructs result of applying `operator` to `arg` upcasting if appropriate.

  Notice `arg` here must be of federated type, with a named tuple member of
  length 2, or a named tuple type of length 2. If the named tuple type of `arg`
  satisfies certain conditions (that is, there is only a single tensor dtype in
  the first element of `arg`, and the second element represents a scalar of
  this dtype), the second element will be upcast to match the first. Here this
  means it will be pushed into a nested structure matching the structure of the
  first element of `arg`. For example, it makes perfect sense to divide a model
  of type `<a=float32[784],b=float32[10]>` by a scalar of type `float32`, but
  the binary operator constructors we have implemented only take arguments of
  type `<T, T>`. Therefore in this case we would broadcast the `float` argument
  to the `tuple` type, before constructing a binary operator which divides
  pointwise.

  Args:
    arg: `federated_language.framework.ComputationBuildingBlock` of federated
      type whose `member` attribute is a named tuple type of length 2, or named
      tuple type of length 2.
    operator: Callable representing binary operator to apply to the 2-tuple
      represented by the federated `arg`.

  Returns:
    Instance of `federated_language.framework.Call`
    encapsulating the result of formally applying `operator` to
    `arg[0], `arg[1]`, upcasting `arg[1]` in the condition described above.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(
      arg, federated_language.framework.ComputationBuildingBlock
  )
  if isinstance(arg.type_signature, federated_language.FederatedType):
    tuple_type = arg.type_signature.member
    assert isinstance(tuple_type, federated_language.StructType)
  elif isinstance(arg.type_signature, federated_language.StructType):
    tuple_type = arg.type_signature
  else:
    raise TypeError(
        'Generic binary operators are only implemented for federated tuple and '
        'unplaced tuples; you have passed {}.'.format(arg.type_signature)
    )

  tf_representing_proto, tf_representing_type = (
      tensorflow_computation_factory.create_binary_operator_with_upcast(
          operator, tuple_type
      )
  )
  tf_representing_op = federated_language.framework.CompiledComputation(
      tf_representing_proto, type_signature=tf_representing_type
  )

  if isinstance(arg.type_signature, federated_language.FederatedType):
    called = federated_language.framework.create_federated_map_or_apply(
        tf_representing_op, arg
    )
  else:
    called = federated_language.framework.Call(tf_representing_op, arg)

  return called
