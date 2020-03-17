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
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
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
) -> trusted_aggregating_executor.TrustedAggregatingExecutor:
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
      trusted_aggregating_executor.AGGREGATOR: bottom_ex,  # FIXME
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


class FederatingExecutorTest(parameterized.TestCase):

  def test_executor_create_value_with_valid_intrinsic_def(self):
    val = _produce_test_value(
        intrinsic_defs.FEDERATED_APPLY,
        computation_types.FunctionType(
            [type_factory.unary_op(tf.int32),
             type_factory.at_server(tf.int32)],
            type_factory.at_server(tf.int32)))
    self.assertIsInstance(
        val, trusted_aggregating_executor.TrustedAggregatingExecutorValue)
    self.assertEqual(
        str(val.type_signature),
        '(<(int32 -> int32),int32@SERVER> -> int32@SERVER)')
    self.assertIs(val.internal_representation, intrinsic_defs.FEDERATED_APPLY)

  def test_executor_create_value_with_invalid_intrinsic_def(self):
    with self.assertRaises(TypeError):
      _produce_test_value(intrinsic_defs.FEDERATED_APPLY, tf.bool)

  def test_executor_create_value_with_intrinsic_as_pb_computation(self):
    val = _produce_test_value(
        pb.Computation(
            intrinsic=pb.Intrinsic(uri='generic_zero'),
            type=type_serialization.serialize_type(tf.int32)))
    self.assertIsInstance(
        val, trusted_aggregating_executor.TrustedAggregatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIs(val.internal_representation, intrinsic_defs.GENERIC_ZERO)

  def test_executor_create_value_with_unbound_reference(self):
    with self.assertRaises(ValueError):
      _produce_test_value(
          pb.Computation(
              reference=pb.Reference(name='a'),
              type=type_serialization.serialize_type(tf.int32)))

  def test_executor_create_value_with_server_int(self):
    val = _produce_test_value(10, type_spec=type_factory.at_server(tf.int32))
    self.assertIsInstance(
        val, trusted_aggregating_executor.TrustedAggregatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@SERVER')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 1)
    self.assertIsInstance(val.internal_representation[0],
                          eager_tf_executor.EagerValue)
    self.assertEqual(
        val.internal_representation[0].internal_representation.numpy(), 10)

  def test_executor_create_value_with_client_int(self):
    val = _produce_test_value([10, 20, 30],
                              type_spec=type_factory.at_clients(tf.int32),
                              num_clients=3)
    self.assertIsInstance(
        val, trusted_aggregating_executor.TrustedAggregatingExecutorValue)
    self.assertEqual(str(val.type_signature), '{int32}@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_tf_executor.EagerValue)
    self.assertCountEqual([
        v.internal_representation.numpy() for v in val.internal_representation
    ], [10, 20, 30])

  def test_executor_create_value_with_all_equal_client_int(self):
    val = _produce_test_value(
        10,
        type_spec=type_factory.at_clients(tf.int32, all_equal=True),
        num_clients=3)
    self.assertIsInstance(
        val, trusted_aggregating_executor.TrustedAggregatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32@CLIENTS')
    self.assertIsInstance(val.internal_representation, list)
    self.assertLen(val.internal_representation, 3)
    for v in val.internal_representation:
      self.assertIsInstance(v, eager_tf_executor.EagerValue)
      self.assertEqual(v.internal_representation.numpy(), 10)

  def test_executor_create_value_with_unplaced_int(self):
    val = _produce_test_value(10, type_spec=tf.int32)
    self.assertIsInstance(
        val, trusted_aggregating_executor.TrustedAggregatingExecutorValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertIsInstance(val.internal_representation,
                          federating_executor.FederatingExecutorValue)
    self.assertIsInstance(val.internal_representation.internal_representation,
                          eager_tf_executor.EagerValue)
    eager_val = val.internal_representation.internal_representation
    self.assertEqual(eager_val.internal_representation.numpy(), 10)


if __name__ == '__main__':
  absltest.main()
