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
"""A strategy for resolving federated types and intrinsics."""

import asyncio
import math

from absl import logging
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_transformations

# Secure Aggregation supports a maximum of 62 bits everywhere. All summands and
# summations will be masked by _atleast_ this value (possibly smaller, if
# requested by the user).
MAXIMUM_SUPPORTED_BITWIDTH = 62


def _extract_numpy_arrays(*args):
  """Extracts the numpy arrays from a structure of tensors."""
  return _map_numpy_or_structure(*args, fn=lambda t: t.numpy())


def _map_numpy_or_structure(*args, fn):
  """Maps a python function to a value.

  Args:
    *args: A list of `tf.Tensor` or `structure.Struct` of tensor arguments to
      apply `fn` to. All arguments must have the same structure, which `fn` is
      mapped pointwise to.
    fn: A Python that takes a single `tf.Tensor` argument.

  Returns:
    A tensor, or structure of tensors, matching the shape of `value`.
  """
  if tf.is_tensor(args[0]) or isinstance(args[0], (np.number, np.ndarray)):
    return fn(*args)
  elif isinstance(args[0], structure.Struct):
    return structure.map_structure(fn, *args)
  else:
    raise TypeError(
        'Received a `value` argument to map with an unknown type: [{t}]. '
        'Only `tf.Tensor` or `structure.Struct` are supported.'.format(
            t=type(args[0])))


def _compute_summation_type_for_bitwidth(bitwidth, type_spec):
  """Creates a `tff.Type` with dtype based on bitwidth."""

  def type_for_bitwidth_limited_tensor(bits, tensor_type):
    if bits < 1 or bits > MAXIMUM_SUPPORTED_BITWIDTH:
      raise ValueError('Encountered an bitwidth that cannot be handled: {b}. '
                       'Extended bitwidth must be between [1,{m}].'
                       '\nRequested: {r}'.format(
                           b=bits, r=bitwidth, m=MAXIMUM_SUPPORTED_BITWIDTH))
    elif bits < 32:
      return computation_types.TensorType(
          shape=tensor_type.shape,
          dtype=tf.uint32 if tensor_type.dtype.is_unsigned else tf.int32)
    else:
      return computation_types.TensorType(
          shape=tensor_type.shape,
          dtype=tf.uint64 if tensor_type.dtype.is_unsigned else tf.int64)

  if type_spec.is_tensor():
    return type_for_bitwidth_limited_tensor(bitwidth, type_spec)
  elif type_spec.is_struct():
    return computation_types.StructType(
        structure.iter_elements(
            structure.map_structure(type_for_bitwidth_limited_tensor, bitwidth,
                                    type_spec)))
  else:
    raise TypeError('Summation types can only be created from TensorType or '
                    'StructType. Received a {t}'.format(t=type_spec))


# IMPORTANT: the TestFederatedStrategy is implemented in a very non-stanrdard
# way, materializing values mid-execution for debug logging, etc. This should
# _NOT_ be an example of how to implement other executors, rather it is better
# to embed the computations fully in the underlying executors and `create_call`
# on those computations.
class TestFederatedStrategy(
    federated_resolving_strategy.FederatedResolvingStrategy):
  """A strategy for resolving federated types and intrinsics.

  This strategy extends the `tff.framework.FederatedResolvingStrategy` and
  provides an insecure implemention of the `tff.federated_secure_sum` intrinsic,
  which can be useful for testing federated algorithms that use this intrinsic.
  """

  @tracing.trace
  async def compute_federated_secure_select(
      self, arg: federated_resolving_strategy.FederatedResolvingStrategyValue
  ) -> federated_resolving_strategy.FederatedResolvingStrategyValue:
    self.compute_federated_select(arg)

  # Note: we intentionally do not cache the result of
  # _compute_extra_bits_for_secagg() to caching failures, and keeping this
  # stateless is easier to reason about. As this is a test backend, we're less
  # concerned with performance and more with correctness.
  async def _compute_extra_bits_for_secagg(self):
    """Compute the number of additional bits required for a secure sum."""
    # First we compute the bitwidth each tensor in the structure will use,
    # padding an extra log2(# of clients) bits for the summand.
    ones = await self._executor.create_value(
        1, computation_types.at_clients(tf.int32, all_equal=True))
    num_clients = await self.compute_federated_sum(ones)
    # Must add log2(# of clients) bits to the bitwidth to ensure the full
    # sum fits inside the mask.
    num_clients = await num_clients.compute()
    logging.debug('Emulating secure sum over %d clients', num_clients.numpy())
    return int(math.ceil(math.log(num_clients.numpy(), 2)))

  async def _embed_tf_secure_sum_mask_value(self, type_spec, extended_bitwidth):
    """Construct a CompiledComputation with the mask for the secure sum."""

    def transform_to_scalar_type_spec(t):
      """Converts all `tff.TensorType` to scalar shapes."""
      if not t.is_tensor():
        return t, False
      return computation_types.TensorType(dtype=t.dtype, shape=[]), True

    mask_type, _ = type_transformations.transform_type_postorder(
        type_spec, transform_to_scalar_type_spec)

    def compute_mask(bits, type_spec):
      mask_value = 2**bits - 1
      if mask_value > type_spec.dtype.max:
        logging.warning(
            'Emulated secure sum mask exceeds maximum value of '
            'dtype. Dtype %s, mask bits: %d', type_spec.dtype, bits)
        mask_value = type_spec.dtype.max
      return tf.constant(mask_value, type_spec.dtype)

    mask_value = _map_numpy_or_structure(
        extended_bitwidth, mask_type, fn=compute_mask)
    logging.debug('Emulated secure sum using mask: %s', mask_value)
    return await executor_utils.embed_constant(self._executor, mask_type,
                                               mask_value)

  async def _compute_modulus(self, value, mask):

    async def build_modulus_argument(value, mask):
      # Create the mask at the same placement as value.
      placed_mask = await self._executor.create_value(
          await mask.compute(),
          computation_types.FederatedType(
              mask.type_signature,
              value.type_signature.placement,
              all_equal=True))
      arg_struct = await self._executor.create_struct([value, placed_mask])
      if value.type_signature.placement == placements.SERVER:
        return await self.compute_federated_zip_at_server(arg_struct)
      elif value.type_signature.placement == placements.CLIENTS:
        return await self.compute_federated_zip_at_clients(arg_struct)
      else:
        raise TypeError(
            'Unknown placement [{p}], must be one of [CLIENTS, SERVER]'.format(
                p=value.type_signature.placement))

    modulus_comp_coro = self._executor.create_value(
        *tensorflow_computation_factory.create_binary_operator_with_upcast(
            computation_types.StructType([
                value.type_signature.member, mask.type_signature
            ]), tf.bitwise.bitwise_and))

    modulus_comp, modulus_comp_arg = await asyncio.gather(
        modulus_comp_coro, build_modulus_argument(value, mask))
    map_arg = federated_resolving_strategy.FederatedResolvingStrategyValue(
        structure.Struct([
            (None, modulus_comp.internal_representation),
            (None, modulus_comp_arg.internal_representation),
        ]),
        computation_types.StructType(
            [modulus_comp.type_signature, modulus_comp_arg.type_signature]))
    if value.type_signature.all_equal:
      return await self.compute_federated_map_all_equal(map_arg)
    else:
      return await self.compute_federated_map(map_arg)

  @tracing.trace
  async def compute_federated_secure_sum(
      self, arg: federated_resolving_strategy.FederatedResolvingStrategyValue
  ) -> federated_resolving_strategy.FederatedResolvingStrategyValue:
    logging.warning(
        'The implementation of the `tff.federated_secure_sum` intrinsic '
        'provided by the `tff.backends.test` runtime uses no cryptography.')
    py_typecheck.check_type(arg.internal_representation, structure.Struct)
    py_typecheck.check_len(arg.internal_representation, 2)
    summands, bitwidth = await asyncio.gather(
        self.ingest_value(arg.internal_representation[0],
                          arg.type_signature[0]).compute(),
        self.ingest_value(arg.internal_representation[1],
                          arg.type_signature[1]).compute())
    summands_type = arg.type_signature[0].member
    if not type_analysis.is_structure_of_integers(summands_type):
      raise TypeError(
          'Cannot compute `federated_secure_sum` on summands that are not '
          'TensorType or StructType of TensorType. Got {t}'.format(
              t=repr(summands_type)))
    if (summands_type.is_struct() and
        not structure.is_same_structure(summands_type, bitwidth)):
      raise TypeError(
          'Cannot compute `federated_secure_sum` if summands and bitwidth are '
          'not the same structure. Got summands={s}, bitwidth={b}'.format(
              s=repr(summands_type), b=repr(bitwidth.type_signature)))

    num_additional_bits = await self._compute_extra_bits_for_secagg()
    # Clamp to 64 bits, otherwise we can't represent the mask in TensorFlow.
    extended_bitwidth = _map_numpy_or_structure(
        bitwidth, fn=lambda b: min(b.numpy() + num_additional_bits, 64))
    logging.debug('Emulated secure sum effective bitwidth: %s',
                  extended_bitwidth)
    # Now we need to cast the summands into the integral type that is large
    # enough to represent the sum and the mask.
    summation_type_spec = _compute_summation_type_for_bitwidth(
        extended_bitwidth, summands_type)
    # `summands` is a list of all clients' summands. We map
    # `_map_numpy_or_structure` to the list, applying it pointwise to clients.
    summand_tensors = tf.nest.map_structure(_extract_numpy_arrays, summands)
    # Dtype conversion trick: pull the summand values out, and push them back
    # into the executor using the new dtypes decided based on bitwidth.
    casted_summands = await self._executor.create_value(
        summand_tensors, computation_types.at_clients(summation_type_spec))
    # To emulate SecAgg without the random masks, we must mask the summands to
    # the effective bitwidth. This isn't strictly necessary because we also
    # mask the sum result and modulus operator is distributive, but this more
    # accurately reflects the system.
    mask = await self._embed_tf_secure_sum_mask_value(summation_type_spec,
                                                      extended_bitwidth)
    masked_summands = await self._compute_modulus(casted_summands, mask)
    logging.debug('Computed masked modular summands as: %s', await
                  masked_summands.compute())
    # Then perform the sum and modolulo operation (using powers of 2 bitmasking)
    # on the sum, using the computed effective bitwidth.
    sum_result = await self.compute_federated_sum(masked_summands)
    modular_sums = await self._compute_modulus(sum_result, mask)
    # Dtype conversion trick again, pull the modular sum values out, and push
    # them back into the executor using the dypte from the summands.
    modular_sum_values = _extract_numpy_arrays(await modular_sums.compute())
    logging.debug('Computed modular sums as: %s', modular_sum_values)
    return await self._executor.create_value(
        modular_sum_values, computation_types.at_server(summands_type))
