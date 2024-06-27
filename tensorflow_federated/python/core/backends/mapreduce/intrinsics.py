# Copyright 2018 Google LLC
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
"""Intrinsics for the mapreduce backend."""

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import symbol_binding_context
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.federated_context import value_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions


# Computes the modular sum of client values on the server, securely. Only
# supported for integers or nested structures of integers.
#
# Type signature: <{V}@CLIENTS,M> -> V@SERVER
FEDERATED_SECURE_MODULAR_SUM = intrinsic_defs.IntrinsicDef(
    'FEDERATED_SECURE_MODULAR_SUM',
    'federated_secure_modular_sum',
    computation_types.FunctionType(
        parameter=[
            computation_types.FederatedType(
                computation_types.AbstractType('V'), placements.CLIENTS
            ),
            computation_types.AbstractType('M'),
        ],
        result=computation_types.FederatedType(
            computation_types.AbstractType('V'), placements.SERVER
        ),
    ),
    aggregation_kind=intrinsic_defs.AggregationKind.SECURE,
)


def _cast(
    comp: building_blocks.ComputationBuildingBlock,
    type_signature: computation_types.TensorType,
) -> building_blocks.Call:
  """Casts `comp` to the provided type."""

  def cast_fn(value):

    def cast_element(element, type_signature: computation_types.TensorType):
      return tf.cast(element, type_signature.dtype)

    if isinstance(comp.type_signature, computation_types.StructType):
      return structure.map_structure(cast_element, value, type_signature)
    return cast_element(value, type_signature)

  cast_proto, cast_type = tensorflow_computation_factory.create_unary_operator(
      cast_fn, comp.type_signature
  )
  cast_comp = building_blocks.CompiledComputation(
      cast_proto, type_signature=cast_type
  )
  return building_blocks.Call(cast_comp, comp)


def create_federated_secure_modular_sum(
    value: building_blocks.ComputationBuildingBlock,
    modulus: building_blocks.ComputationBuildingBlock,
    preapply_modulus: bool = True,
) -> building_blocks.ComputationBuildingBlock:
  r"""Creates a called secure modular sum.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    modulus: A `building_blocks.ComputationBuildingBlock` to use as the
      `modulus` value.
    preapply_modulus: Whether or not to preapply `modulus` to the input `value`.
      This can be `False` if `value` is guaranteed to already be in range.

  Returns:
    A computation building block which invokes `federated_secure_modular_sum`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(modulus, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(
      value.type_signature.member,  # pytype: disable=attribute-error
      placements.SERVER,
  )
  intrinsic_type = computation_types.FunctionType(
      [
          type_conversions.type_to_non_all_equal(value.type_signature),
          modulus.type_signature,
      ],
      result_type,
  )
  intrinsic = building_blocks.Intrinsic(
      FEDERATED_SECURE_MODULAR_SUM.uri, intrinsic_type
  )

  if not preapply_modulus:
    values = building_blocks.Struct([value, modulus])
    return building_blocks.Call(intrinsic, values)

  # Pre-insert a modulus to ensure the the input values are within range.
  mod_ref = building_blocks.Reference('mod', modulus.type_signature)

  # In order to run `tf.math.floormod`, our modulus and value must be the same
  # type.
  casted_mod = _cast(
      mod_ref,
      value.type_signature.member,  # pytype: disable=attribute-error
  )
  # Since in the preapply_modulus case the modulus is expected to be available
  # at the client as well as at the server for aggregation, we need to broadcast
  # the modulus to be able to avoid repeating the modulus value (which could
  # cause accuracy issues if the modulus is non-deterministic).
  casted_mod_at_server = building_block_factory.create_federated_value(
      casted_mod, placements.SERVER
  )
  value_with_mod = building_block_factory.create_federated_zip(
      building_blocks.Struct([
          value,
          building_block_factory.create_federated_broadcast(
              casted_mod_at_server
          ),
      ])
  )

  def structural_modulus(value, mod):
    return structure.map_structure(tf.math.floormod, value, mod)

  structural_modulus_proto, structural_modulus_type = (
      tensorflow_computation_factory.create_binary_operator(
          structural_modulus,
          value.type_signature.member,  # pytype: disable=attribute-error
          casted_mod.type_signature,
      )
  )
  structural_modulus_tf = building_blocks.CompiledComputation(
      structural_modulus_proto, type_signature=structural_modulus_type
  )
  value_modded = building_block_factory.create_federated_map_or_apply(
      structural_modulus_tf, value_with_mod
  )
  values = building_blocks.Struct([value_modded, mod_ref])
  return building_blocks.Block(
      [('mod', modulus)], building_blocks.Call(intrinsic, values)
  )


def create_null_federated_secure_modular_sum():
  return create_federated_secure_modular_sum(
      building_block_factory.create_federated_value(
          building_blocks.Struct([]), placements.CLIENTS
      ),
      building_blocks.Struct([]),
      preapply_modulus=False,
  )


def _bind_comp_as_reference(comp):
  context = context_stack_impl.context_stack.current
  if not isinstance(context, symbol_binding_context.SymbolBindingContext):
    raise context_base.ContextError(
        f'Attempted to construct an intrinsic in context {context} which '
        ' does not support binding references.'
    )
  return context.bind_computation_to_reference(comp)


def federated_secure_modular_sum(value, modulus):
  """Computes a modular sum at `tff.SERVER` of a `value` from `tff.CLIENTS`.

  This function computes a sum such that it should not be possible for the
  server to learn any clients individual value. The specific algorithm and
  mechanism used to compute the secure sum may vary depending on the target
  runtime environment the computation is compiled for or executed on. See
  https://research.google/pubs/pub47246/ for more information.

  Not all executors support
  `tff.backends.mapreduce.federated_secure_modular_sum()`; consult the
  documentation for the specific executor or executor stack you plan on using
  for the specific of how it's handled by that executor.

  The `modulus` argument is the modulus under which the client values are added.
  The result of this function will be equivalent to `SUM(value) % modulus`.
  *Lower values may allow for improved communication efficiency.*

  Example:

  ```python
  value = tff.federated_value(5, tff.CLIENTS)
  result = tff.backends.mapreduce.federated_secure_modular_sum(value, 3)
  # `result == (5 * num_clients % 3)@SERVER`

  value = tff.federated_value((3, 9), tff.CLIENTS)
  result = tff.backends.mapreduce.federated_secure_modular_sum(
      value, (100, 200))
  # `result == (3 * num_clients % 100, 9 * num_clients % 100)@SERVER`
  ```

  Note: To sum non-integer values or to sum integers with fewer constraints and
  weaker privacy properties, consider using `federated_sum`.

  Args:
    value: An integer or nested structure of integers placed at `tff.CLIENTS`.
      Values outside of the range [0, modulus-1] will be considered equivalent
      to mod(value, modulus), i.e. they will be projected into the range [0,
      modulus-1] as part of the modular summation.
    modulus: A Python integer or nested structure of integers matching the
      structure of `value`. If integer `modulus` is used with a nested `value`,
      the same integer is used for each tensor in `value`.

  Returns:
    A representation of the modular sum of the member constituents of `value`
    placed on the `tff.SERVER`.  The resulting modular sum will be on the range
    [0, modulus-1].

  Raises:
    TypeError: If the argument is not a federated TFF value placed at
      `tff.CLIENTS`.
  """
  value = value_impl.to_value(value, type_spec=None)
  value = value_utils.ensure_federated_value(
      value, placements.CLIENTS, 'value to be summed'
  )
  type_analysis.check_is_structure_of_integers(value.type_signature)
  modulus_value = value_impl.to_value(modulus, type_spec=None)
  value_member_type = value.type_signature.member  # pytype: disable=attribute-error
  modulus_type = modulus_value.type_signature
  if not type_analysis.is_single_integer_or_matches_structure(
      modulus_type, value_member_type
  ):
    raise TypeError(
        'Expected `federated_secure_sum` parameter `modulus` to match '
        'the structure of `value`, with one integer max per tensor in '
        '`value`. Found `value` of `{}` and `modulus` of `{}`.'.format(
            value_member_type, modulus_type
        )
    )
  if isinstance(modulus_type, computation_types.TensorType) and isinstance(
      value_member_type, computation_types.StructType
  ):
    modulus_value = value_impl.to_value(
        structure.map_structure(lambda _: modulus, value_member_type),
        type_spec=None,
    )
  comp = create_federated_secure_modular_sum(value.comp, modulus_value.comp)
  comp = _bind_comp_as_reference(comp)
  return value_impl.Value(comp)
