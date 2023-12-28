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

from typing import Optional
import unittest
from unittest import mock
import uuid

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import data_source as data_source_lib
from tensorflow_federated.python.program import prefetching_data_source


class _TestDataSourceIterator(data_source_lib.FederatedDataSourceIterator):
  """A test implementation of `tff.program.FederatedDataSourceIterator`.

  A `tff.program.ProgramStateManager` cannot be constructed directly because it
  has abstract methods, this implementation exists to make it possible to
  construct instances of `tff.program.ProgramStateManager` that can used as
  stubs or mocked.
  """

  def __init__(self):
    self._uuid = uuid.uuid4()

  @classmethod
  def from_bytes(cls, buffer: bytes) -> '_TestDataSourceIterator':
    instance = _TestDataSourceIterator()
    instance._uuid = uuid.UUID(bytes=buffer)
    return instance

  def to_bytes(self) -> bytes:
    uuid_bytes = self._uuid.bytes
    return uuid_bytes

  @property
  def federated_type(self) -> computation_types.FederatedType:
    return computation_types.FederatedType(np.int32, placements.CLIENTS)

  def select(self, k: Optional[int] = None) -> object:
    return [1, 2, 3]

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, _TestDataSourceIterator):
      return NotImplemented
    return self._uuid == other._uuid


class PrefetchingDataSourceIteratorTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_does_not_raise_type_error(self):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )

    try:
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          total_iterations=5,
          iterations_to_prefetch=3,
          k_to_prefetch=3,
          prefetch_threshold=1,
      )
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_iterator(self, iterator):

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=iterator,
          total_iterations=5,
          iterations_to_prefetch=3,
          k_to_prefetch=3,
          prefetch_threshold=1,
      )

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_total_iterations(self, total_iterations):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          total_iterations=total_iterations,
          iterations_to_prefetch=3,
          k_to_prefetch=3,
          prefetch_threshold=1,
      )

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_iterations_to_prefetch(
      self, iterations_to_prefetch
  ):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          total_iterations=5,
          iterations_to_prefetch=iterations_to_prefetch,
          k_to_prefetch=3,
          prefetch_threshold=1,
      )

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_k_to_prefetch(self, k_to_prefetch):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          total_iterations=5,
          iterations_to_prefetch=10,
          k_to_prefetch=k_to_prefetch,
          prefetch_threshold=1,
      )

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_prefetch_threshold(
      self, prefetch_threshold
  ):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          total_iterations=5,
          iterations_to_prefetch=10,
          k_to_prefetch=3,
          prefetch_threshold=prefetch_threshold,
      )

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -1),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_value_error_with_k_to_prefetch(self, k_to_prefetch):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )

    with self.assertRaises(ValueError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          total_iterations=5,
          iterations_to_prefetch=3,
          k_to_prefetch=k_to_prefetch,
          prefetch_threshold=1,
      )

  @context_stack_test_utils.with_context(
      execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_init_raises_runtime_error_with_context(self):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )

    with self.assertRaises(RuntimeError):
      prefetching_data_source.PrefetchingDataSourceIterator(
          iterator=mock_iterator,
          total_iterations=5,
          iterations_to_prefetch=3,
          k_to_prefetch=3,
          prefetch_threshold=1,
      )

  @parameterized.named_parameters(
      ('one', 1),
      ('two', 2),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  async def test_select_returns_data_with_k(self, k):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )
    mock_iterator.select.return_value = list(range(k))
    mock_iterator.federated_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        total_iterations=5,
        iterations_to_prefetch=3,
        k_to_prefetch=k,  # Must be the same.
        prefetch_threshold=1,
    )

    data = iterator.select(k)

    @federated_computation.federated_computation(iterator.federated_type)
    def _identity(value):
      return value

    actual_value = await _identity(data)
    expected_value = list(range(k))
    self.assertEqual(actual_value, expected_value)

  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_prefetches_data(self):
    iterations_to_prefetch = 3
    k_to_prefetch = 5

    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )
    mock_iterator.select.return_value = [1, 2, 3, 4, 5]
    mock_iterator.federated_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        total_iterations=5,
        iterations_to_prefetch=iterations_to_prefetch,
        k_to_prefetch=k_to_prefetch,
        prefetch_threshold=1,
    )

    # Should prefetch data worth iterations_to_prefetch, making one select
    # call to the underlying data source per iteration (in parallel).
    self.assertEqual(iterator._iterations_prefetched, iterations_to_prefetch)

    # Finish the async fetching of data, and verify the prefetched cache.
    iterator._finish_prefetching()
    self.assertEqual(mock_iterator.select.call_count, iterations_to_prefetch)
    self.assertLen(iterator._prefetched_data, iterations_to_prefetch)

  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_data_is_prefetched_when_at_or_below_threshold(self):
    iterations_to_prefetch = 3
    k_to_prefetch = 5

    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )
    mock_iterator.select.return_value = [1, 2, 3, 4, 5]
    mock_iterator.federated_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        total_iterations=10,
        iterations_to_prefetch=iterations_to_prefetch,
        k_to_prefetch=k_to_prefetch,
        prefetch_threshold=2,
    )
    expected_iterations_prefetched = iterations_to_prefetch
    self.assertEqual(
        iterator._iterations_prefetched, expected_iterations_prefetched
    )

    # Selecting one iteration worth of data, would drop the cache below
    # threshold i.e. 2, and trigger a replenishing of the cache.
    expected_iterations_prefetched = expected_iterations_prefetched + 1
    iterator.select(k_to_prefetch)
    self.assertEqual(
        iterator._iterations_prefetched, expected_iterations_prefetched
    )

    # Finish the async fetching of data, and verify the prefetched cache.
    iterator._finish_prefetching()
    self.assertEqual(
        mock_iterator.select.call_count, expected_iterations_prefetched
    )
    self.assertLen(iterator._prefetched_data, iterations_to_prefetch)

  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_data_is_not_prefetched_when_not_below_threshold(self):
    iterations_to_prefetch = 3
    k_to_prefetch = 5

    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )
    mock_iterator.select.return_value = [1, 2, 3, 4, 5]
    mock_iterator.federated_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        total_iterations=10,
        iterations_to_prefetch=iterations_to_prefetch,
        k_to_prefetch=k_to_prefetch,
        prefetch_threshold=1,
    )
    expected_iterations_prefetched = iterations_to_prefetch
    expected_prefetched_data_size = iterations_to_prefetch
    self.assertEqual(
        iterator._iterations_prefetched, expected_iterations_prefetched
    )
    iterator._finish_prefetching()
    self.assertLen(iterator._prefetched_data, expected_prefetched_data_size)
    self.assertEqual(
        mock_iterator.select.call_count, expected_iterations_prefetched
    )

    # Selecting one itertation worth of data, would still leave enough
    # iterations of data in the prefetched cache, so no new prefetches would be
    # triggered.
    expected_prefetched_data_size -= 1
    iterator.select(k_to_prefetch)
    self.assertEqual(
        iterator._iterations_prefetched, expected_iterations_prefetched
    )  # No additional calls.

    # Finish the async fetching of data, and verify the prefetched cache.
    iterator._finish_prefetching()
    self.assertEqual(
        mock_iterator.select.call_count, expected_iterations_prefetched
    )  # No additional calls.
    self.assertLen(iterator._prefetched_data, expected_prefetched_data_size)

  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_data_is_prefeched_only_for_iterations_remaining(self):
    iterations_to_prefetch = 3
    k_to_prefetch = 5
    total_iterations = 2

    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )
    mock_iterator.select.return_value = [1, 2, 3, 4, 5]
    mock_iterator.federated_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        total_iterations=total_iterations,
        iterations_to_prefetch=iterations_to_prefetch,
        k_to_prefetch=k_to_prefetch,
        prefetch_threshold=2,
    )

    # Only 2 iterations worth of data should be prefetched, though
    # iterations_to_prefetch is 3.
    expected_iterations_prefetched = total_iterations
    expected_prefetched_data_size = total_iterations
    self.assertEqual(
        iterator._iterations_prefetched, expected_iterations_prefetched
    )

    # Finish the async fetching of data, and verify the prefetched cache.
    iterator._finish_prefetching()
    self.assertEqual(
        mock_iterator.select.call_count, expected_iterations_prefetched
    )
    self.assertLen(iterator._prefetched_data, expected_prefetched_data_size)

    # Selecting one iteration of data should use up one iteration from the
    # prefetched cache but should not trigger any additional prefetches.
    expected_prefetched_data_size -= 1
    iterator.select(k_to_prefetch)
    self.assertEqual(
        mock_iterator.select.call_count, expected_iterations_prefetched
    )
    self.assertLen(iterator._prefetched_data, expected_prefetched_data_size)

  @parameterized.named_parameters(
      ('str', 'a'),
      ('list', []),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_select_raises_type_error_with_k(self, k):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        total_iterations=5,
        iterations_to_prefetch=3,
        k_to_prefetch=3,
        prefetch_threshold=1,
    )

    with self.assertRaises(TypeError):
      iterator.select(k)

  @parameterized.named_parameters(
      ('none', None),
      ('negative', -1),
      ('different', 4),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_select_raises_value_error_with_k(self, k):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        total_iterations=5,
        iterations_to_prefetch=3,
        k_to_prefetch=3,
        prefetch_threshold=1,
    )

    with self.assertRaises(ValueError):
      iterator.select(k)

  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_select_raises_runtime_error_with_too_many_iterations(self):
    mock_iterator = mock.create_autospec(
        data_source_lib.FederatedDataSourceIterator,
        spec_set=True,
        instance=True,
    )
    mock_iterator.select.return_value = [1, 2, 3]
    mock_iterator.federated_type = computation_types.FederatedType(
        np.int32, placements.CLIENTS
    )
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=mock_iterator,
        total_iterations=5,
        iterations_to_prefetch=3,
        k_to_prefetch=3,
        prefetch_threshold=1,
    )

    for _ in range(5):
      iterator.select(k=3)

    with self.assertRaises(RuntimeError):
      iterator.select(k=3)

  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_serializable(self):
    stub_iterator = _TestDataSourceIterator()
    iterator = prefetching_data_source.PrefetchingDataSourceIterator(
        iterator=stub_iterator,
        total_iterations=5,
        iterations_to_prefetch=3,
        k_to_prefetch=3,
        prefetch_threshold=1,
    )
    iterator_bytes = iterator.to_bytes()

    actual_iterator = (
        prefetching_data_source.PrefetchingDataSourceIterator.from_bytes(
            iterator_bytes
        )
    )

    self.assertIsNot(actual_iterator, iterator)
    self.assertEqual(actual_iterator, iterator)


class PrefetchingDataSourceTest(parameterized.TestCase):

  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_does_not_raise_type_error(self):
    mock_data_source = mock.create_autospec(
        data_source_lib.FederatedDataSource, spec_set=True, instance=True
    )

    try:
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          total_iterations=5,
          iterations_to_prefetch=3,
          k_to_prefetch=3,
          prefetch_threshold=1,
      )
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_data_source(self, data_source):

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=data_source,
          total_iterations=5,
          iterations_to_prefetch=3,
          k_to_prefetch=3,
          prefetch_threshold=1,
      )

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_total_iterations(self, total_iterations):
    mock_data_source = mock.create_autospec(
        data_source_lib.FederatedDataSource, spec_set=True, instance=True
    )

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          total_iterations=total_iterations,
          iterations_to_prefetch=3,
          k_to_prefetch=3,
          prefetch_threshold=1,
      )

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_iterations_to_prefetch(
      self, iterations_to_prefetch
  ):
    mock_data_source = mock.create_autospec(
        data_source_lib.FederatedDataSource, spec_set=True, instance=True
    )

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          total_iterations=5,
          iterations_to_prefetch=iterations_to_prefetch,
          k_to_prefetch=3,
          prefetch_threshold=1,
      )

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_k_to_prefetch(self, k_to_prefetch):
    mock_data_source = mock.create_autospec(
        data_source_lib.FederatedDataSource, spec_set=True, instance=True
    )

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          total_iterations=5,
          iterations_to_prefetch=3,
          k_to_prefetch=k_to_prefetch,
          prefetch_threshold=1,
      )

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_type_error_with_prefetch_threshold(
      self, prefetch_threshold
  ):
    mock_data_source = mock.create_autospec(
        data_source_lib.FederatedDataSource, spec_set=True, instance=True
    )

    with self.assertRaises(TypeError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          total_iterations=5,
          iterations_to_prefetch=3,
          k_to_prefetch=3,
          prefetch_threshold=prefetch_threshold,
      )

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -1),
  )
  @context_stack_test_utils.with_context(
      execution_contexts.create_async_local_cpp_execution_context
  )
  def test_init_raises_value_error_with_k_to_prefetch(self, k_to_prefetch):
    mock_data_source = mock.create_autospec(
        data_source_lib.FederatedDataSource, spec_set=True, instance=True
    )

    with self.assertRaises(ValueError):
      prefetching_data_source.PrefetchingDataSource(
          data_source=mock_data_source,
          total_iterations=5,
          iterations_to_prefetch=3,
          k_to_prefetch=k_to_prefetch,
          prefetch_threshold=1,
      )


if __name__ == '__main__':
  absltest.main()
