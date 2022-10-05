# Copyright 2022, The TensorFlow Federated Authors.
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
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import dataset_data_source
from tensorflow_federated.python.program import prefetching_data_source


class PrefetchingDataSourceTest(parameterized.TestCase,
                                unittest.IsolatedAsyncioTestCase):

  @parameterized.named_parameters(
      ('int', [1, 2, 3], tf.int32),
      ('str', ['a', 'b', 'c'], tf.string),
  )
  def test_init_sets_federated_type(self, tensors, dtype):
    datasets = [tf.data.Dataset.from_tensor_slices(tensors)] * 3
    ds = dataset_data_source.DatasetDataSource(datasets)
    context = execution_contexts.create_local_async_python_execution_context()

    prefetching_ds = prefetching_data_source.PrefetchingDataSource(
        data_source=ds,
        total_rounds=5,
        num_rounds_to_prefetch=3,
        num_clients_to_prefetch=2,
        context=context,
        buffer_size=0)

    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(dtype), placements.CLIENTS)
    self.assertEqual(prefetching_ds.federated_type, federated_type)

  async def test_select_calls_prefetches_data(self):
    ds = mock.create_autospec(data_source.FederatedDataSource, autospec=True)
    ds_iterator = mock.create_autospec(
        data_source.FederatedDataSourceIterator, autospec=True)
    ds_iterator.federated_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS)
    ds_iterator.select.return_value = [1, 2]
    ds.iterator.return_value = ds_iterator
    context = execution_contexts.create_local_async_python_execution_context()

    prefetching_ds = prefetching_data_source.PrefetchingDataSource(
        data_source=ds,
        total_rounds=5,
        num_rounds_to_prefetch=3,
        num_clients_to_prefetch=2,
        context=context,
        buffer_size=0)

    ds_iterator.select.assert_not_called()
    # The first round of prefetching
    prefetching_iter = prefetching_ds.iterator()
    for _ in range(2):
      prefetching_iter.select(number_of_clients=2)
      self.assertEqual(ds_iterator.select.call_count, 3)
    prefetching_iter.select(number_of_clients=2)
    # The second round of prefetching
    await asyncio.sleep(0.1)
    self.assertEqual(ds_iterator.select.call_count, 5)
    for _ in range(2):
      prefetching_iter.select(number_of_clients=2)
      self.assertEqual(ds_iterator.select.call_count, 5)
    with self.assertRaises(RuntimeError):
      prefetching_iter.select(number_of_clients=2)


if __name__ == '__main__':
  absltest.main()
