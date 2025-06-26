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

import federated_language
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_computation_factory


# Computes the modular sum of client values on the server, securely. Only
# supported for integers or nested structures of integers.
#
# Type signature: <{V}@CLIENTS,M> -> V@SERVER
FEDERATED_SECURE_MODULAR_SUM = federated_language.framework.IntrinsicDef(
    'FEDERATED_SECURE_MODULAR_SUM',
    'federated_secure_modular_sum',
    federated_language.FunctionType(
        parameter=[
            federated_language.FederatedType(
                federated_language.AbstractType('V'), federated_language.CLIENTS
            ),
            federated_language.AbstractType('M'),
        ],
        result=federated_language.FederatedType(
            federated_language.AbstractType('V'), federated_language.SERVER
        ),
    ),
    aggregation_kind=federated_language.framework.AggregationKind.SECURE,
)


def _cast(
    comp: federated_language.framework.ComputationBuildingBlock,
    type_signature: federated_language.TensorType,
) -> federated_language.framework.Call:
  """Casts `comp` to the provided type."""

  def cast_fn(value):

    def cast_element(element, type_signature: federated_language.TensorType):
      return tf.cast(element, type_signature.dtype)

    if isinstance(comp.type_signature, federated_language.StructType):
      return structure._map_structure(cast_element, value, type_signature)  # pylint: disable=protected-access
    return cast_element(value, type_signature)

  cast_proto, cast_type = tensorflow_computation_factory.create_unary_operator(
      cast_fn, comp.type_signature
  )
  cast_comp = federated_language.framework.CompiledComputation(
      cast_proto, type_signature=cast_type
  )
  return federated_language.framework.Call(cast_comp, comp)


def create_federated_secure_modular_sum(
    value: federated_language.framework.ComputationBuildingBlock,
    modulus: federated_language.framework.ComputationBuildingBlock,
    preapply_modulus: bool = True,
) -> federated_language.framework.ComputationBuildingBlock:
  r"""Creates a called secure modular sum.

  Args:
    value: A `federated_language.framework.ComputationBuildingBlock` to use as
      the value.
    modulus: A `federated_language.framework.ComputationBuildingBlock` to use as
      the `modulus` value.
    preapply_modulus: Whether or not to preapply `modulus` to the input `value`.
      This can be `False` if `value` is guaranteed to already be in range.

  Returns:
    A computation building block which invokes `federated_secure_modular_sum`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(
      value, federated_language.framework.ComputationBuildingBlock
  )
  py_typecheck.check_type(
      modulus, federated_language.framework.ComputationBuildingBlock
  )

  if not isinstance(value.type_signature, federated_language.FederatedType):
    raise ValueError(
        'Expected `value.type_signature` to be a'
        f' `federated_language.FederatedType`, found {value.type_signature}'
    )
  value_type = federated_language.FederatedType(
      value.type_signature.member,
      value.type_signature.placement,
      all_equal=False,
  )

  result_type = federated_language.FederatedType(
      value.type_signature.member,  # pytype: disable=attribute-error
      federated_language.SERVER,
  )
  intrinsic_type = federated_language.FunctionType(
      [value_type, modulus.type_signature], result_type
  )
  intrinsic = federated_language.framework.Intrinsic(
      FEDERATED_SECURE_MODULAR_SUM.uri, intrinsic_type
  )

  if not preapply_modulus:
    values = federated_language.framework.Struct([value, modulus])
    return federated_language.framework.Call(intrinsic, values)

  # Pre-insert a modulus to ensure the the input values are within range.
  mod_ref = federated_language.framework.Reference(
      'mod', modulus.type_signature
  )

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
  casted_mod_at_server = federated_language.framework.create_federated_value(
      casted_mod, federated_language.SERVER
  )
  value_with_mod = federated_language.framework.create_federated_zip(
      federated_language.framework.Struct([
          value,
          federated_language.framework.create_federated_broadcast(
              casted_mod_at_server
          ),
      ])
  )

  def structural_modulus(value, mod):
    return structure._map_structure(tf.math.floormod, value, mod)  # pylint: disable=protected-access

  structural_modulus_proto, structural_modulus_type = (
      tensorflow_computation_factory.create_binary_operator(
          structural_modulus,
          value.type_signature.member,  # pytype: disable=attribute-error
          casted_mod.type_signature,
      )
  )
  structural_modulus_tf = federated_language.framework.CompiledComputation(
      structural_modulus_proto, type_signature=structural_modulus_type
  )
  value_modded = federated_language.framework.create_federated_map_or_apply(
      structural_modulus_tf, value_with_mod
  )
  values = federated_language.framework.Struct([value_modded, mod_ref])
  return federated_language.framework.Block(
      [('mod', modulus)], federated_language.framework.Call(intrinsic, values)
  )


def create_null_federated_secure_modular_sum():
  return create_federated_secure_modular_sum(
      federated_language.framework.create_federated_value(
          federated_language.framework.Struct([]), federated_language.CLIENTS
      ),
      federated_language.framework.Struct([]),
      preapply_modulus=False,
  )


def _bind_comp_as_reference(comp):
  context = federated_language.framework.get_context_stack().current
  if not isinstance(context, federated_language.framework.SymbolBindingContext):
    raise federated_language.framework.ContextError(
        f'Attempted to construct an intrinsic in context {context} which '
        ' does not support binding references.'
    )
  return context.bind_computation_to_reference(comp)


def federated_secure_modular_sum(value, modulus):
  """Computes a modular sum at `federated_language.SERVER` of a `value` from `federated_language.CLIENTS`.

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
  value = federated_language.federated_value(5, federated_language.CLIENTS)
  result = tff.backends.mapreduce.federated_secure_modular_sum(value, 3)
  # `result == (5 * num_clients % 3)@SERVER`

  value = federated_language.federated_value((3, 9), federated_language.CLIENTS)
  result = tff.backends.mapreduce.federated_secure_modular_sum(
      value, (100, 200))
  # `result == (3 * num_clients % 100, 9 * num_clients % 100)@SERVER`
  ```

  Note: To sum non-integer values or to sum integers with fewer constraints and
  weaker privacy properties, consider using `federated_sum`.

  Args:
    value: An integer or nested structure of integers placed at
      `federated_language.CLIENTS`. Values outside of the range [0, modulus-1]
      will be considered equivalent to mod(value, modulus), i.e. they will be
      projected into the range [0, modulus-1] as part of the modular summation.
    modulus: A Python integer or nested structure of integers matching the
      structure of `value`. If integer `modulus` is used with a nested `value`,
      the same integer is used for each tensor in `value`.

  Returns:
    A representation of the modular sum of the member constituents of `value`
    placed on the `federated_language.SERVER`.  The resulting modular sum will
    be on the range
    [0, modulus-1].

  Raises:
    TypeError: If the argument is not a federated TFF value placed at
      `federated_language.CLIENTS`.
  """
  value = federated_language.to_value(value, type_spec=None)
  value = federated_language.framework.ensure_federated_value(
      value, federated_language.CLIENTS, 'value to be summed'
  )
  federated_language.framework.check_is_structure_of_integers(
      value.type_signature
  )
  modulus_value = federated_language.to_value(modulus, type_spec=None)
  value_member_type = value.type_signature.member  # pytype: disable=attribute-error
  modulus_type = modulus_value.type_signature
  if not federated_language.framework.is_single_integer_or_matches_structure(
      modulus_type, value_member_type
  ):
    raise TypeError(
        'Expected `federated_secure_sum` parameter `modulus` to match '
        'the structure of `value`, with one integer max per tensor in '
        '`value`. Found `value` of `{}` and `modulus` of `{}`.'.format(
            value_member_type, modulus_type
        )
    )
  if isinstance(modulus_type, federated_language.TensorType) and isinstance(
      value_member_type, federated_language.StructType
  ):
    modulus_value = federated_language.to_value(
        structure._map_structure(lambda _: modulus, value_member_type),  # pylint: disable=protected-access
        type_spec=None,
    )
  comp = create_federated_secure_modular_sum(value.comp, modulus_value.comp)
  comp = _bind_comp_as_reference(comp)
  return federated_language.Value(comp)
