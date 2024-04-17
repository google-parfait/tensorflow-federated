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

import collections

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


class ExecutionContextIntegrationTest(parameterized.TestCase):

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_simple_no_arg_tf_computation_with_int_result(self):

    @tff.tensorflow.computation
    def comp():
      return tf.constant(10)

    result = comp()

    self.assertEqual(result, 10)

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_one_arg_tf_computation_with_int_param_and_result(self):

    @tff.tensorflow.computation(np.int32)
    def comp(x):
      return tf.add(x, 10)

    result = comp(3)

    self.assertEqual(result, 13)

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_three_arg_tf_computation_with_int_params_and_result(self):

    @tff.tensorflow.computation(np.int32, np.int32, np.int32)
    def comp(x, y, z):
      return tf.multiply(tf.add(x, y), z)

    result = comp(3, 4, 5)

    self.assertEqual(result, 35)

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_tf_computation_with_dataset_params_and_int_result(self):

    @tff.tensorflow.computation(tff.SequenceType(np.int32))
    def comp(ds):
      return ds.reduce(np.int32(0), lambda x, y: x + y)

    ds = tf.data.Dataset.range(10).map(lambda x: tf.cast(x, np.int32))
    result = comp(ds)

    self.assertEqual(result, 45)

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_tf_computation_with_structured_result(self):

    @tff.tensorflow.computation
    def comp():
      return collections.OrderedDict([
          ('a', tf.constant(10)),
          ('b', tf.constant(20)),
      ])

    result = comp()

    self.assertIsInstance(result, collections.OrderedDict)
    self.assertDictEqual(result, {'a': 10, 'b': 20})

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_changing_cardinalities_across_calls(self):

    @tff.federated_computation(tff.FederatedType(np.int32, tff.CLIENTS))
    def comp(x):
      return x

    five_ints = list(range(5))
    ten_ints = list(range(10))

    five = comp(five_ints)
    ten = comp(ten_ints)

    self.assertEqual(five, five_ints)
    self.assertEqual(ten, ten_ints)

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_conflicting_cardinalities_within_call(self):

    @tff.federated_computation([
        tff.FederatedType(np.int32, tff.CLIENTS),
        tff.FederatedType(np.int32, tff.CLIENTS),
    ])
    def comp(x):
      return x

    five_ints = list(range(5))
    ten_ints = list(range(10))

    with self.assertRaisesRegex(ValueError, 'Conflicting cardinalities'):
      comp([five_ints, ten_ints])

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_tuple_argument_can_accept_unnamed_elements(self):

    @tff.tensorflow.computation(np.int32, np.int32)
    def foo(x, y):
      return x + y

    result = foo(tff.structure.Struct([(None, 2), (None, 3)]))  # pylint: disable=no-value-for-parameter

    self.assertEqual(result, 5)


if __name__ == '__main__':
  absltest.main()
