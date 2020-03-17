# Lint as: python3
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

import asyncio
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.compiler import type_serialization
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.executors import trusted_aggregating_executor

tf.compat.v1.enable_v2_behavior()


def _make_test_executor(
    num_clients=1,
    use_reference_resolving_executor=False,
) -> trusted_aggregating_Executor.TrustedAggregatingExecutor:
  bottom_ex = eager_tf_executor.EagerTFExecutor()
  if use_reference_resolving_executor:
    bottom_ex = reference_resolving_executor.ReferenceResolvingExecutor(
        bottom_ex)
  fed_targets = {
      placements.SERVER: bottom_ex,
      placements.CLIENTS: [bottom_ex for _ in range(num_clients)],
      None: bottom_ex
  }
  fed_ex = federating_executor.FederatingExecutor(fed_targets)
  aggr_targets = {
      "AGGREGATOR": bottom_ex,
      None: fed_ex
  }
  aggr_targets = {**fed_targets, **aggr_targets}
  return trusted_aggregating_executor.TrustedAggregatingExecutor(aggr_targets)


Runtime = Tuple[asyncio.AbstractEventLoop,
                federating_executor.FederatingExecutor]


def _make_test_runtime(
    num_clients=1, use_reference_resolving_executor=False) -> Runtime:
  """Creates a test runtime consisting of an event loop and test executor."""
  loop = asyncio.get_event_loop()
  ex = _make_test_executor(
      num_clients=num_clients,
      use_reference_resolving_executor=use_reference_resolving_executor)
  return loop, ex


def _run_comp_with_runtime(comp, runtime: Runtime):
  """Runs a computation using the provided runtime."""
  loop, ex = runtime

  async def call_value():
    return await ex.create_call(await ex.create_value(comp))

  return loop.run_until_complete(call_value())


def _run_test_comp(comp, num_clients=1, use_reference_resolving_executor=False):
  """Runs a computation (unapplied TFF function) using a test runtime."""
  runtime = _make_test_runtime(
      num_clients=num_clients,
      use_reference_resolving_executor=use_reference_resolving_executor)
  return _run_comp_with_runtime(comp, runtime)


def _run_test_comp_produces_federated_value(
    test_instance,
    comp,
    num_clients=1,
    use_reference_resolving_executor=False,
):
  """Runs a computation (unapplied TFF function) using a test runtime.

  This is similar to _run_test_comp, but the result is asserted to be a
  FederatedValue and computed.

  Args:
    test_instance: A class with the standard unit testing assertions.
    comp: The computation to run.
    num_clients: The number of clients to use when computing `comp`.
    use_reference_resolving_executor: Whether or not to include an executor
      to resolve references.

  Returns:
    The result of running the computation.
  """
  loop, ex = _make_test_runtime(
      num_clients=num_clients,
      use_reference_resolving_executor=use_reference_resolving_executor)
  val = _run_comp_with_runtime(comp, (loop, ex))
  test_instance.assertIsInstance(val,
                                 federating_executor.FederatingExecutorValue)
  return loop.run_until_complete(val.compute())


def _produce_test_value(
    value,
    type_spec=None,
    num_clients=1,
    use_reference_resolving_executor=False,
):
  """Produces a TFF value using a test runtime."""
  loop, ex = _make_test_runtime(
      num_clients=num_clients,
      use_reference_resolving_executor=use_reference_resolving_executor)
  return loop.run_until_complete(ex.create_value(value, type_spec=type_spec))
