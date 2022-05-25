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

import asyncio
import collections

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.tensorflow_libs import tensorflow_test_utils


class DatasetsTest(parameterized.TestCase):
  """Tests for Datasets in a native backend.

  These tests ensure that `tf.data.Datasets`s are passed through to TF without
  TFF mutating or changing the data.
  """

  def setUp(self):
    super().setUp()
    execution_contexts.set_local_python_execution_context()

  @tensorflow_test_utils.skip_test_for_gpu
  def test_takes_dataset(self):

    @tensorflow_computation.tf_computation
    def foo(ds):
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    ds = tf.data.Dataset.range(10)
    actual_result = foo(ds)

    expected_result = ds.take(10).reduce(np.int64(0), lambda x, y: x + y)
    self.assertEqual(actual_result, expected_result)

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_dataset(self):

    @tensorflow_computation.tf_computation
    def foo():
      return tf.data.Dataset.range(10)

    actual_result = foo()

    expected_result = tf.data.Dataset.range(10)
    self.assertEqual(
        list(actual_result.as_numpy_iterator()),
        list(expected_result.as_numpy_iterator()))

  def test_takes_dataset_infinite(self):

    @tensorflow_computation.tf_computation
    def foo(ds):
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    ds = tf.data.Dataset.range(10).repeat()
    actual_result = foo(ds)

    expected_result = ds.take(10).reduce(np.int64(0), lambda x, y: x + y)
    self.assertEqual(actual_result, expected_result)

  def test_returns_dataset_infinite(self):

    @tensorflow_computation.tf_computation
    def foo():
      return tf.data.Dataset.range(10).repeat()

    actual_result = foo()

    expected_result = tf.data.Dataset.range(10).repeat()
    self.assertEqual(
        actual_result.take(100).reduce(np.int64(0), lambda x, y: x + y),
        expected_result.take(100).reduce(np.int64(0), lambda x, y: x + y))

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_dataset_two(self):

    @tensorflow_computation.tf_computation
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

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_dataset_and_tensor(self):

    @tensorflow_computation.tf_computation
    def foo():
      return [tf.data.Dataset.range(5), tf.constant(5)]

    actual_result = foo()

    expected_result = [tf.data.Dataset.range(5), tf.constant(5)]
    self.assertEqual(
        list(actual_result[0].as_numpy_iterator()),
        list(expected_result[0].as_numpy_iterator()))
    self.assertEqual(actual_result[1], expected_result[1])

  @tensorflow_test_utils.skip_test_for_gpu
  def test_returns_empty_dataset(self):

    @tensorflow_computation.tf_computation
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


class AsyncLocalExecutionContextTest(absltest.TestCase):
  """Ensures that the context in the tested file integrates with asyncio."""

  def setUp(self):
    super().setUp()
    execution_contexts.set_local_async_python_execution_context()

  def test_single_coro_invocation(self):

    @tensorflow_computation.tf_computation
    def return_one():
      return 1

    result = asyncio.run(return_one())
    self.assertEqual(result, 1)

  def test_asyncio_gather(self):

    @tensorflow_computation.tf_computation
    def return_one():
      return 1

    @tensorflow_computation.tf_computation
    def return_two():
      return 2

    async def await_comps():
      return await asyncio.gather(return_one(), return_two())

    result = asyncio.run(await_comps())
    self.assertEqual(result, [1, 2])


if __name__ == '__main__':
  absltest.main()
