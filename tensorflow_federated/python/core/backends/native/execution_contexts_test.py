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

import collections

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test as common_test
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.native import execution_contexts


class DatasetsTest(parameterized.TestCase):
  """Tests for Datasets in a native backend.

  These tests ensure that `tf.data.Datasets`s are passed through to TF without
  TFF mutating or changing the data.
  """

  @common_test.skip_test_for_gpu
  def test_takes_dataset(self):

    @computations.tf_computation
    def foo(ds):
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    ds = tf.data.Dataset.range(10)
    actual_result = foo(ds)

    expected_result = ds.take(10).reduce(np.int64(0), lambda x, y: x + y)
    self.assertEqual(actual_result, expected_result)

  @common_test.skip_test_for_gpu
  def test_returns_dataset(self):

    @computations.tf_computation
    def foo():
      return tf.data.Dataset.range(10)

    actual_result = foo()

    expected_result = tf.data.Dataset.range(10)
    self.assertEqual(
        list(actual_result.as_numpy_iterator()),
        list(expected_result.as_numpy_iterator()))

  def test_takes_dataset_infinite(self):

    @computations.tf_computation
    def foo(ds):
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    ds = tf.data.Dataset.range(10).repeat()
    actual_result = foo(ds)

    expected_result = ds.take(10).reduce(np.int64(0), lambda x, y: x + y)
    self.assertEqual(actual_result, expected_result)

  def test_returns_dataset_infinite(self):

    @computations.tf_computation
    def foo():
      return tf.data.Dataset.range(10).repeat()

    actual_result = foo()

    expected_result = tf.data.Dataset.range(10).repeat()
    self.assertEqual(
        actual_result.take(100).reduce(np.int64(0), lambda x, y: x + y),
        expected_result.take(100).reduce(np.int64(0), lambda x, y: x + y))

  @common_test.skip_test_for_gpu
  def test_returns_dataset_two(self):

    @computations.tf_computation
    def foo():
      return [tf.data.Dataset.range(5), tf.data.Dataset.range(10)]

    actual_result = foo()

    expected_result = [tf.data.Dataset.range(5), tf.data.Dataset.range(10)]
    self.assertEqual(
        list(actual_result[0].as_numpy_iterator()),
        list(expected_result[0].as_numpy_iterator()))
    self.assertEqual(
        list(actual_result[1].as_numpy_iterator()),
        list(expected_result[1].as_numpy_iterator()))

  @common_test.skip_test_for_gpu
  def test_returns_dataset_and_tensor(self):

    @computations.tf_computation
    def foo():
      return [tf.data.Dataset.range(5), tf.constant(5)]

    actual_result = foo()

    expected_result = [tf.data.Dataset.range(5), tf.constant(5)]
    self.assertEqual(
        list(actual_result[0].as_numpy_iterator()),
        list(expected_result[0].as_numpy_iterator()))
    self.assertEqual(actual_result[1], expected_result[1])

  @common_test.skip_test_for_gpu
  def test_returns_empty_dataset(self):

    @computations.tf_computation
    def foo():
      tensor_slices = collections.OrderedDict([('a', [1, 1]), ('b', [1, 1])])
      ds = tf.data.Dataset.from_tensor_slices(tensor_slices)
      return ds.batch(5).take(0)

    actual_result = foo()

    expected_element_spec = collections.OrderedDict([
        ('a', tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        ('b', tf.TensorSpec(shape=(None,), dtype=tf.int32)),
    ])
    self.assertEqual(actual_result.element_spec, expected_element_spec)
    expected_result = tf.data.Dataset.range(10).batch(5).take(0)
    self.assertEqual(
        list(actual_result.as_numpy_iterator()),
        list(expected_result.as_numpy_iterator()))


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  absltest.main()
