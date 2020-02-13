# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections

import attr
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import executor_stacks
from tensorflow_federated.python.core.impl.context_stack import set_default_executor
from tensorflow_federated.python.core.utils import computation_utils

tf.compat.v1.enable_v2_behavior()


class ComputationUtilsTest(test.TestCase):

  def test_update_state_namedtuple(self):
    my_tuple_type = collections.namedtuple('my_tuple_type', 'a b c')
    state = my_tuple_type(1, 2, 3)
    state2 = computation_utils.update_state(state, c=7)
    self.assertEqual(state2, my_tuple_type(1, 2, 7))
    state3 = computation_utils.update_state(state2, a=8)
    self.assertEqual(state3, my_tuple_type(8, 2, 7))

  def test_update_state_dict(self):
    state = collections.OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    state2 = computation_utils.update_state(state, c=7)
    self.assertEqual(state2, {'a': 1, 'b': 2, 'c': 7})
    state3 = computation_utils.update_state(state2, a=8)
    self.assertEqual(state3, {'a': 8, 'b': 2, 'c': 7})

  def test_update_state_ordereddict(self):
    state = collections.OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    state2 = computation_utils.update_state(state, c=7)
    self.assertEqual(state2,
                     collections.OrderedDict([('a', 1), ('b', 2), ('c', 7)]))
    state3 = computation_utils.update_state(state2, a=8)
    self.assertEqual(state3,
                     collections.OrderedDict([('a', 8), ('b', 2), ('c', 7)]))

  def test_update_state_attrs(self):

    @attr.s
    class TestAttrsClass(object):
      a = attr.ib()
      b = attr.ib()
      c = attr.ib()

    state = TestAttrsClass(1, 2, 3)
    state2 = computation_utils.update_state(state, c=7)
    self.assertEqual(state2, TestAttrsClass(1, 2, 7))
    state3 = computation_utils.update_state(state2, a=8)
    self.assertEqual(state3, TestAttrsClass(8, 2, 7))

  def test_update_state_fails(self):
    with self.assertRaisesRegex(TypeError, 'state must be a structure'):
      computation_utils.update_state((1, 2, 3), a=8)
    with self.assertRaisesRegex(TypeError, 'state must be a structure'):
      computation_utils.update_state([1, 2, 3], a=8)
    with self.assertRaisesRegex(KeyError, 'does not contain a field'):
      computation_utils.update_state({'z': 1}, a=8)


def broadcast_initialize_fn():
  return collections.OrderedDict([('call_count', 0)])


def broadcast_next_fn(state, value):

  @computations.tf_computation(tf.int32)
  def add_one(value):
    return value + 1

  return intrinsics.federated_zip(
      collections.OrderedDict([
          ('call_count', intrinsics.federated_map(add_one, state.call_count))
      ])), intrinsics.federated_broadcast(value)


class StatefulBroadcastFnTest(test.TestCase):

  def test_execute(self):
    broadcast_fn = computation_utils.StatefulBroadcastFn(
        initialize_fn=broadcast_initialize_fn, next_fn=broadcast_next_fn)
    broadcast_arg_type = computation_types.FederatedType(
        tf.float32, placements.SERVER)

    @computations.federated_computation(broadcast_arg_type)
    def federated_broadcast_test(args):
      state = intrinsics.federated_value(broadcast_fn.initialize(),
                                         placements.SERVER)
      return broadcast_fn(state, args)

    expected_type_signature = computation_types.FunctionType(
        parameter=broadcast_arg_type,
        result=computation_types.NamedTupleType([
            computation_types.FederatedType(
                collections.OrderedDict([('call_count', tf.int32)]),
                placements.SERVER),
            computation_types.FederatedType(
                tf.float32, placements.CLIENTS, all_equal=True)
        ]))
    self.assertEqual(federated_broadcast_test.type_signature,
                     expected_type_signature)
    state, value = federated_broadcast_test(1.0)
    self.assertAlmostEqual(value, 1.0)
    self.assertDictEqual(state._asdict(), {'call_count': 1})


def agg_initialize_fn():
  return collections.OrderedDict([('call_count', 0)])


def agg_next_fn(state, value, weight):

  @computations.tf_computation(tf.int32)
  def add_one(value):
    return value + 1

  return intrinsics.federated_zip(
      collections.OrderedDict([
          ('call_count', intrinsics.federated_map(add_one, state.call_count))
      ])), intrinsics.federated_mean(value, weight)


class StatefulAggregateFnTest(test.TestCase):

  def test_execute_with_default_weight(self):
    aggregate_fn = computation_utils.StatefulAggregateFn(
        initialize_fn=agg_initialize_fn, next_fn=agg_next_fn)
    aggregate_arg_type = computation_types.FederatedType(
        tf.float32, placements.CLIENTS)

    @computations.federated_computation(aggregate_arg_type)
    def federated_aggregate_test(args):
      state = intrinsics.federated_value(aggregate_fn.initialize(),
                                         placements.SERVER)
      return aggregate_fn(state, args)

    expected_type_signature = computation_types.FunctionType(
        parameter=aggregate_arg_type,
        result=computation_types.NamedTupleType([
            computation_types.FederatedType(
                collections.OrderedDict([('call_count', tf.int32)]),
                placements.SERVER),
            computation_types.FederatedType(tf.float32, placements.SERVER)
        ]))
    self.assertEqual(federated_aggregate_test.type_signature,
                     expected_type_signature)
    state, mean = federated_aggregate_test([1.0, 2.0, 3.0])
    self.assertAlmostEqual(mean, 2.0)  # (1 + 2 + 3) / (1 + 1 + 1)
    self.assertDictEqual(state._asdict(), {'call_count': 1})

  def test_execute_with_explicit_weights(self):
    aggregate_fn = computation_utils.StatefulAggregateFn(
        initialize_fn=agg_initialize_fn, next_fn=agg_next_fn)

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS),
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def federated_aggregate_test(args, weights):
      state = intrinsics.federated_value(aggregate_fn.initialize(),
                                         placements.SERVER)
      return aggregate_fn(state, args, weights)

    state, mean = federated_aggregate_test([1.0, 2.0, 3.0], [4.0, 1.0, 1.0])
    self.assertAlmostEqual(mean, 1.5)  # (1*4 + 2*1 + 3*1) / (4 + 1 + 1)
    self.assertDictEqual(state._asdict(), {'call_count': 1})


if __name__ == '__main__':
  # NOTE: num_clients must be explicit here to correctly test the broadcast
  # behavior. Otherwise TFF will infer there are zero clients, which is an
  # error.
  set_default_executor.set_default_executor(
      executor_stacks.local_executor_factory(num_clients=3))
  test.main()
