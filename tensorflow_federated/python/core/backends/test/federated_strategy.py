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

from absl import logging
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.types import placement_literals


class TestFederatedStrategy(
    federated_resolving_strategy.FederatedResolvingStrategy):
  """A strategy for resolving federated types and intrinsics.

  This strategy extends the `tff.framework.FederatedResolvingStrategy` and
  provides an insecure implemention of the `tff.federated_secure_sum` intrinsic,
  which can be useful for testing federated algorithms that use this intrinsic.
  """

  @tracing.trace
  async def compute_federated_secure_sum(
      self, arg: federated_resolving_strategy.FederatedResolvingStrategyValue
  ) -> federated_resolving_strategy.FederatedResolvingStrategyValue:
    py_typecheck.check_type(arg.internal_representation, structure.Struct)
    py_typecheck.check_len(arg.internal_representation, 2)
    logging.warning(
        'The implementation of the `tff.federated_secure_sum` intrinsic '
        'provided by the `tff.backends.test` runtime uses no cryptography.')

    server_ex = self._target_executors[placement_literals.SERVER][0]
    value = federated_resolving_strategy.FederatedResolvingStrategyValue(
        arg.internal_representation[0], arg.type_signature[0])
    bitwidth = await arg.internal_representation[1].compute()
    bitwidth_type = arg.type_signature[1]
    sum_result, mask, fn = await asyncio.gather(
        self.compute_federated_sum(value),
        executor_utils.embed_tf_scalar_constant(self._executor, bitwidth_type,
                                                2**bitwidth - 1),
        executor_utils.embed_tf_binary_operator(self._executor, bitwidth_type,
                                                tf.math.mod))
    fn_arg = await server_ex.create_struct([
        sum_result.internal_representation[0],
        mask.internal_representation,
    ])
    fn_arg_type = computation_types.FederatedType(
        fn_arg.type_signature, placement_literals.SERVER, all_equal=True)
    arg = federated_resolving_strategy.FederatedResolvingStrategyValue(
        structure.Struct([
            (None, fn.internal_representation),
            (None, [fn_arg]),
        ]), computation_types.StructType([fn.type_signature, fn_arg_type]))
    return await self.compute_federated_map_all_equal(arg)
