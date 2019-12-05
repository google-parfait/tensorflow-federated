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

from absl.testing import absltest
import attr
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import values
from tensorflow_federated.python.core.utils import computation_utils


# Create two tff.Computations that perform sum on a sequence: initializes the
# state to 0 and add each item in a sequence to the state.
@computations.tf_computation
def initialize():
  return tf.constant(0)


@computations.tf_computation([tf.int32, tf.int32])
def add_int32(current, val):
  return current + val


@computations.tf_computation([tf.int32, tf.int32])
def add_mul_int32(current, val):
  return current + val, current * val


@computations.tf_computation(tf.int32)
def count_int32(current):
  return current + 1


class ComputationUtilsTest(absltest.TestCase):

  def test_update_state_namedtuple(self):
    my_tuple_type = collections.namedtuple('my_tuple_type', 'a b c')
    state = my_tuple_type(1, 2, 3)
    state2 = computation_utils.update_state(state, c=7)
    self.assertEqual(state2, my_tuple_type(1, 2, 7))
    state3 = computation_utils.update_state(state2, a=8)
    self.assertEqual(state3, my_tuple_type(8, 2, 7))

  def test_update_state_dict(self):
    state = {'a': 1, 'b': 2, 'c': 3}
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

  def test_iterative_process_state_only(self):
    iterative_process = computation_utils.IterativeProcess(
        initialize, count_int32)

    state = iterative_process.initialize()
    iterations = 10
    for _ in range(iterations):
      # TODO(b/122321354): remove the .item() call on `state` once numpy.int32
      # type is supported.
      state = iterative_process.next(state.item())
    self.assertEqual(state, iterations)

  def test_iterative_process_state_tuple_arg(self):
    iterative_process = computation_utils.IterativeProcess(
        initialize, add_int32)

    state = iterative_process.initialize()
    iterations = 10
    for val in range(iterations):
      state = iterative_process.next(state, val)
    self.assertEqual(state, sum(range(iterations)))

  def test_iterative_process_state_multiple_return_values(self):
    iterative_process = computation_utils.IterativeProcess(
        initialize, add_mul_int32)

    state = iterative_process.initialize()
    iterations = 10
    for val in range(iterations):
      state, product = iterative_process.next(state, val)
    self.assertEqual(state, sum(range(iterations)))
    self.assertEqual(product, sum(range(iterations - 1)) * (iterations - 1))

  def test_iterative_process_initialize_bad_type(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      _ = computation_utils.IterativeProcess(
          initialize_fn=None, next_fn=add_int32)

    with self.assertRaisesRegex(
        TypeError, r'initialize_fn must be a no-arg tff.Computation'):

      @computations.federated_computation(tf.int32)
      def one_arg_initialize(one_arg):
        del one_arg  # unused
        return values.to_value(0)

      _ = computation_utils.IterativeProcess(
          initialize_fn=one_arg_initialize, next_fn=add_int32)

  def test_iterative_process_next_bad_type(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      _ = computation_utils.IterativeProcess(
          initialize_fn=initialize, next_fn=None)

  def test_iterative_process_type_mismatch(self):
    with self.assertRaisesRegex(
        TypeError, r'The return type of initialize_fn should match.*'):

      @computations.federated_computation([tf.float32, tf.float32])
      def add_float32(current, val):
        return current + val

      _ = computation_utils.IterativeProcess(
          initialize_fn=initialize, next_fn=add_float32)

    with self.assertRaisesRegex(
        TypeError,
        'The return type of next_fn should match the first parameter'):

      @computations.federated_computation(tf.int32)
      def add_bad_result(_):
        return 0.0

      _ = computation_utils.IterativeProcess(
          initialize_fn=initialize, next_fn=add_bad_result)

    with self.assertRaisesRegex(
        TypeError,
        'The return type of next_fn should match the first parameter'):

      @computations.federated_computation(tf.int32)
      def add_bad_multi_result(_):
        return 0.0, 0

      _ = computation_utils.IterativeProcess(
          initialize_fn=initialize, next_fn=add_bad_multi_result)


def broadcast_initialize_fn():
  return {'call_count': 0}


def broadcast_next_fn(state, value):

  @computations.tf_computation(tf.int32)
  def add_one(value):
    return value + 1

  return {
      'call_count': intrinsics.federated_map(add_one, state.call_count),
  }, intrinsics.federated_broadcast(value)


class StatefulBroadcastFnTest(absltest.TestCase):

  def test_construct_with_default_weight(self):

    @computations.federated_computation(
        computation_types.FederatedType(
            tf.float32, placements.SERVER, all_equal=True))
    def federated_broadcast_test(args):
      broadcast_fn = computation_utils.StatefulBroadcastFn(
          initialize_fn=broadcast_initialize_fn, next_fn=broadcast_next_fn)
      state = intrinsics.federated_value(broadcast_fn.initialize(),
                                         placements.SERVER)
      return broadcast_fn(state, args)

    state, value = federated_broadcast_test(1.0)
    self.assertAlmostEqual(value, 1.0)
    self.assertDictEqual(state._asdict(), {'call_count': 1})


def agg_initialize_fn():
  return {'call_count': 0}


def agg_next_fn(state, value, weight):

  @computations.tf_computation(tf.int32)
  def add_one(value):
    return value + 1

  return {
      'call_count': intrinsics.federated_map(add_one, state.call_count),
  }, intrinsics.federated_mean(value, weight)


class StatefulAggregateFnTest(absltest.TestCase):

  def test_construct_with_default_weight(self):

    @computations.federated_computation(
        computation_types.FederatedType(
            tf.float32, placements.CLIENTS, all_equal=False))
    def federated_aggregate_test(args):
      aggregate_fn = computation_utils.StatefulAggregateFn(
          initialize_fn=agg_initialize_fn, next_fn=agg_next_fn)
      state = intrinsics.federated_value(aggregate_fn.initialize(),
                                         placements.SERVER)
      return aggregate_fn(state, args)

    state, mean = federated_aggregate_test([1.0, 2.0, 3.0])
    self.assertAlmostEqual(mean, 2.0)  # (1 + 2 + 3) / (1 + 1 + 1)
    self.assertDictEqual(state._asdict(), {'call_count': 1})

  def test_construct_with_explicit_weights(self):

    @computations.federated_computation(
        computation_types.FederatedType(
            tf.float32, placements.CLIENTS, all_equal=False),
        computation_types.FederatedType(
            tf.float32, placements.CLIENTS, all_equal=False))
    def federated_aggregate_test(args, weights):
      aggregate_fn = computation_utils.StatefulAggregateFn(
          initialize_fn=agg_initialize_fn, next_fn=agg_next_fn)
      state = intrinsics.federated_value(aggregate_fn.initialize(),
                                         placements.SERVER)
      return aggregate_fn(state, args, weights)

    state, mean = federated_aggregate_test([1.0, 2.0, 3.0], [4.0, 1.0, 1.0])
    self.assertAlmostEqual(mean, 1.5)  # (1*4 + 2*1 + 3*1) / (4 + 1 + 1)
    self.assertDictEqual(state._asdict(), {'call_count': 1})


if __name__ == '__main__':
  absltest.main()
