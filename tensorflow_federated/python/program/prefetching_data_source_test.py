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
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import data_source as data_source_lib
from tensorflow_federated.python.program import prefetching_data_source


class PrefetchingDataSourceIteratorTest(parameterized.TestCase):

  def test_init_does_not_raise_type_error(self):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)
    context = execution_contexts.create_local_async_python_execution_context()

    try:
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_iterator(self, iterator):
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=iterator,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)

  # pyformat: disable
  @parameterized.named_parameters(
      ('async_python',
       execution_contexts.create_local_async_python_execution_context()),
  )
  # pyformat: enable
  def test_init_does_not_raise_type_error_with_context(self, context):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)

    try:
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  # pyformat: disable
  @parameterized.named_parameters(
      ('sync_cpp',
       execution_contexts.create_local_python_execution_context()),
      ('sync_python',
       execution_contexts.create_local_python_execution_context()),
  )
  # pyformat: enable
  def test_init_raises_type_error_with_context(self, context):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_total_rounds(self, total_rounds):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          context=context,
          total_rounds=total_rounds,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_num_rounds_to_prefetch(
      self, num_rounds_to_prefetch):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=num_rounds_to_prefetch,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_num_clients_to_prefetch(
      self, num_clients_to_prefetch):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=10,
          num_clients_to_prefetch=num_clients_to_prefetch,
          prefetch_threshold=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_prefetch_threshold(
      self, prefetch_threshold):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=10,
          num_clients_to_prefetch=3,
          prefetch_threshold=prefetch_threshold)

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -1),
  )
  def test_init_raises_value_error_with_num_clients_to_prefetch(
      self, num_clients_to_prefetch):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(ValueError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=num_clients_to_prefetch,
          prefetch_threshold=1)

  @parameterized.named_parameters(
      ('str', 'a'),
      ('list', []),
  )
  def test_select_raises_type_error_with_num_clients(self, num_clients):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)
    context = execution_contexts.create_local_async_python_execution_context()
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        context=context,
        total_rounds=5,
        num_rounds_to_prefetch=3,
        num_clients_to_prefetch=3,
        prefetch_threshold=1)

    with self.assertRaises(TypeError):
      iterator.select(num_clients)

  @parameterized.named_parameters(
      ('none', None),
      ('negative', -1),
      ('different', 4),
  )
  def test_select_raises_value_error_with_num_clients(self, num_clients):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)
    context = execution_contexts.create_local_async_python_execution_context()
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        context=context,
        total_rounds=5,
        num_rounds_to_prefetch=3,
        num_clients_to_prefetch=3,
        prefetch_threshold=1)

    with self.assertRaises(ValueError):
      iterator.select(num_clients)


class PrefetchingDataSourceTest(parameterized.TestCase,
                                unittest.IsolatedAsyncioTestCase):

  def test_init_does_not_raise_type_error(self):
    mock_data_source = mock.create_autospec(data_source_lib.FederatedDataSource)
    context = execution_contexts.create_local_async_python_execution_context()

    try:
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_data_source(self, data_source):
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=data_source,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)

  # pyformat: disable
  @parameterized.named_parameters(
      ('async_python',
       execution_contexts.create_local_async_python_execution_context()),
  )
  # pyformat: enable
  def test_init_does_not_raise_type_error_with_context(self, context):
    mock_data_source = mock.create_autospec(data_source_lib.FederatedDataSource)

    try:
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  # pyformat: disable
  @parameterized.named_parameters(
      ('sync_cpp',
       execution_contexts.create_local_python_execution_context()),
      ('sync_python',
       execution_contexts.create_local_python_execution_context()),
  )
  # pyformat: enable
  def test_init_raises_type_error_with_context(self, context):
    mock_data_source = mock.create_autospec(data_source_lib.FederatedDataSource)

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_total_rounds(self, total_rounds):
    mock_data_source = mock.create_autospec(data_source_lib.FederatedDataSource)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          context=context,
          total_rounds=total_rounds,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_num_rounds_to_prefetch(
      self, num_rounds_to_prefetch):
    mock_data_source = mock.create_autospec(data_source_lib.FederatedDataSource)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=num_rounds_to_prefetch,
          num_clients_to_prefetch=3,
          prefetch_threshold=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_num_clients_to_prefetch(
      self, num_clients_to_prefetch):
    mock_data_source = mock.create_autospec(data_source_lib.FederatedDataSource)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=num_clients_to_prefetch,
          prefetch_threshold=1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_prefetch_threshold(
      self, prefetch_threshold):
    mock_data_source = mock.create_autospec(data_source_lib.FederatedDataSource)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=3,
          prefetch_threshold=prefetch_threshold)

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -1),
  )
  def test_init_raises_value_error_with_num_clients_to_prefetch(
      self, num_clients_to_prefetch):
    mock_data_source = mock.create_autospec(data_source_lib.FederatedDataSource)
    context = execution_contexts.create_local_async_python_execution_context()

    with self.assertRaises(ValueError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          context=context,
          total_rounds=5,
          num_rounds_to_prefetch=3,
          num_clients_to_prefetch=num_clients_to_prefetch,
          prefetch_threshold=1)

  async def test_select_calls_prefetches_data(self):
    ds = mock.create_autospec(data_source_lib.FederatedDataSource)
    ds_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator)
    ds_iterator.federated_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS)
    ds_iterator.select.return_value = [1, 2]
    ds.iterator.return_value = ds_iterator
    context = execution_contexts.create_local_async_python_execution_context()

    prefetching_ds = prefetching_data_source.PrefetchingDataSource(
        data_source=ds,
        context=context,
        total_rounds=5,
        num_rounds_to_prefetch=3,
        num_clients_to_prefetch=2,
        prefetch_threshold=0)

    ds_iterator.select.assert_not_called()
    # The first round of prefetching
    prefetching_iter = prefetching_ds.iterator()
    for _ in range(2):
      prefetching_iter.select(num_clients=2)
      self.assertEqual(ds_iterator.select.call_count, 3)
    prefetching_iter.select(num_clients=2)
    # The second round of prefetching
    await asyncio.sleep(0.1)
    self.assertEqual(ds_iterator.select.call_count, 5)
    for _ in range(2):
      prefetching_iter.select(num_clients=2)
      self.assertEqual(ds_iterator.select.call_count, 5)
    with self.assertRaises(RuntimeError):
      prefetching_iter.select(num_clients=2)


if __name__ == '__main__':
  absltest.main()
