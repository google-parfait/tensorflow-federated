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
import collections
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import federated_context
from tensorflow_federated.python.program import native_platform


async def _coro(value: Any) -> Any:
  return value


class AwaitableValueReferenceTest(parameterized.TestCase,
                                  unittest.IsolatedAsyncioTestCase):

  @parameterized.named_parameters(
      ('tensor', _coro(1), computation_types.TensorType(tf.int32)),
      ('sequence', _coro([1, 2, 3]), computation_types.SequenceType(tf.int32)),
  )
  def test_init_does_not_raise_type_error_with_type_signature(
      self, awaitable, type_signature):

    try:
      native_platform.AwaitableValueReference(
          awaitable=awaitable, type_signature=type_signature)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_init_raises_type_error_with_awaitable(self, awaitable):
    type_signature = computation_types.TensorType(tf.int32)

    with self.assertRaises(TypeError):
      native_platform.AwaitableValueReference(
          awaitable=awaitable, type_signature=type_signature)

  @parameterized.named_parameters(
      ('federated', computation_types.FederatedType(tf.int32,
                                                    placements.SERVER)),
      ('struct', computation_types.StructWithPythonType([], list)),
  )
  def test_init_raises_type_error_with_type_signature(self, type_signature):
    awaitable = _coro(1)

    with self.assertRaises(TypeError):
      native_platform.AwaitableValueReference(
          awaitable=awaitable, type_signature=type_signature)

  @parameterized.named_parameters(
      ('bool', _coro(True), computation_types.TensorType(tf.bool), True),
      ('int', _coro(1), computation_types.TensorType(tf.int32), 1),
      ('str', _coro('a'), computation_types.TensorType(tf.string), 'a'),
  )
  async def test_get_value_returns_value(self, awaitable, type_signature,
                                         expected_value):
    value_reference = native_platform.AwaitableValueReference(
        awaitable=awaitable, type_signature=type_signature)

    actual_value = await value_reference.get_value()

    self.assertEqual(actual_value, expected_value)


class CreateStructureOfAwaitableReferencesTest(parameterized.TestCase,
                                               unittest.IsolatedAsyncioTestCase
                                              ):

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor',
       _coro(1),
       computation_types.TensorType(tf.int32),
       native_platform.AwaitableValueReference(
           _coro(1), computation_types.TensorType(tf.int32))),
      ('federated',
       _coro(1),
       computation_types.FederatedType(tf.int32, placements.SERVER),
       native_platform.AwaitableValueReference(
           _coro(1), computation_types.TensorType(tf.int32))),
      ('struct_unnamed',
       _coro([True, 1, 'a']),
       computation_types.StructWithPythonType([
           tf.bool, tf.int32, tf.string], list),
       structure.Struct([
           (None, native_platform.AwaitableValueReference(
               _coro(True), computation_types.TensorType(tf.bool))),
           (None, native_platform.AwaitableValueReference(
               _coro(1), computation_types.TensorType(tf.int32))),
           (None, native_platform.AwaitableValueReference(
               _coro('a'), computation_types.TensorType(tf.string))),
       ])),
      ('struct_named',
       _coro(collections.OrderedDict([('a', True), ('b', 1), ('c', 'a')])),
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
       ], collections.OrderedDict),
       structure.Struct([
           ('a', native_platform.AwaitableValueReference(
               _coro(True), computation_types.TensorType(tf.bool))),
           ('b', native_platform.AwaitableValueReference(
               _coro(1), computation_types.TensorType(tf.int32))),
           ('c', native_platform.AwaitableValueReference(
               _coro('a'), computation_types.TensorType(tf.string))),
       ])),
      ('struct_nested',
       _coro(collections.OrderedDict([
           ('x', collections.OrderedDict([('a', True), ('b', 1)])),
           ('y', collections.OrderedDict([('c', 'a')])),
       ])),
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', tf.bool),
               ('b', tf.int32),
           ], collections.OrderedDict)),
           ('y', computation_types.StructWithPythonType([
               ('c', tf.string),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
       structure.Struct([
           ('x', structure.Struct([
               ('a', native_platform.AwaitableValueReference(
                   _coro(True), computation_types.TensorType(tf.bool))),
               ('b', native_platform.AwaitableValueReference(
                   _coro(1), computation_types.TensorType(tf.int32))),
           ])),
           ('y', structure.Struct([
               ('c', native_platform.AwaitableValueReference(
                   _coro('a'), computation_types.TensorType(tf.string))),
           ])),
       ])),
  )
  # pyformat: enable
  async def test_returns_value_materialized_sequentially(
      self, awaitable, type_signature, expected_value):
    actual_value = native_platform._create_structure_of_awaitable_references(
        awaitable=awaitable, type_signature=type_signature)

    if (type_signature.is_struct() and
        not structure.is_same_structure(actual_value, expected_value)):
      self.fail('Expected the structures to be the same, found '
                f'{actual_value} and {expected_value}')
    actual_flattened = structure.flatten(actual_value)
    actual_materialized = [await v.get_value() for v in actual_flattened]
    expected_flattened = structure.flatten(expected_value)
    expected_materialized = [await v.get_value() for v in expected_flattened]
    self.assertEqual(actual_materialized, expected_materialized)

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor',
       _coro(1),
       computation_types.TensorType(tf.int32),
       native_platform.AwaitableValueReference(
           _coro(1), computation_types.TensorType(tf.int32))),
      ('federated',
       _coro(1),
       computation_types.FederatedType(tf.int32, placements.SERVER),
       native_platform.AwaitableValueReference(
           _coro(1), computation_types.TensorType(tf.int32))),
      ('struct_unnamed',
       _coro([True, 1, 'a']),
       computation_types.StructWithPythonType([
           tf.bool, tf.int32, tf.string], list),
       structure.Struct([
           (None, native_platform.AwaitableValueReference(
               _coro(True), computation_types.TensorType(tf.bool))),
           (None, native_platform.AwaitableValueReference(
               _coro(1), computation_types.TensorType(tf.int32))),
           (None, native_platform.AwaitableValueReference(
               _coro('a'), computation_types.TensorType(tf.string))),
       ])),
      ('struct_named',
       _coro(collections.OrderedDict([('a', True), ('b', 1), ('c', 'a')])),
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
       ], collections.OrderedDict),
       structure.Struct([
           ('a', native_platform.AwaitableValueReference(
               _coro(True), computation_types.TensorType(tf.bool))),
           ('b', native_platform.AwaitableValueReference(
               _coro(1), computation_types.TensorType(tf.int32))),
           ('c', native_platform.AwaitableValueReference(
               _coro('a'), computation_types.TensorType(tf.string))),
       ])),
      ('struct_nested',
       _coro(collections.OrderedDict([
           ('x', collections.OrderedDict([('a', True), ('b', 1)])),
           ('y', collections.OrderedDict([('c', 'a')])),
       ])),
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', tf.bool),
               ('b', tf.int32),
           ], collections.OrderedDict)),
           ('y', computation_types.StructWithPythonType([
               ('c', tf.string),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
       structure.Struct([
           ('x', structure.Struct([
               ('a', native_platform.AwaitableValueReference(
                   _coro(True), computation_types.TensorType(tf.bool))),
               ('b', native_platform.AwaitableValueReference(
                   _coro(1), computation_types.TensorType(tf.int32))),
           ])),
           ('y', structure.Struct([
               ('c', native_platform.AwaitableValueReference(
                   _coro('a'), computation_types.TensorType(tf.string))),
           ])),
       ])),
  )
  # pyformat: enable
  async def test_returns_value_materialized_concurrently(
      self, awaitable, type_signature, expected_value):
    actual_value = native_platform._create_structure_of_awaitable_references(
        awaitable=awaitable, type_signature=type_signature)

    if (type_signature.is_struct() and
        not structure.is_same_structure(actual_value, expected_value)):
      self.fail('Expected the structures to be the same, found '
                f'{actual_value} and {expected_value}')
    actual_flattened = structure.flatten(actual_value)
    actual_materialized = await asyncio.gather(
        *[v.get_value() for v in actual_flattened])
    expected_flattened = structure.flatten(expected_value)
    expected_materialized = await asyncio.gather(
        *[v.get_value() for v in expected_flattened])
    self.assertEqual(actual_materialized, expected_materialized)

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor',
       _coro(1),
       computation_types.TensorType(tf.int32),
       native_platform.AwaitableValueReference(
           _coro(1), computation_types.TensorType(tf.int32))),
      ('federated',
       _coro(1),
       computation_types.FederatedType(tf.int32, placements.SERVER),
       native_platform.AwaitableValueReference(
           _coro(1), computation_types.TensorType(tf.int32))),
      ('struct_unnamed',
       _coro([True, 1, 'a']),
       computation_types.StructWithPythonType([
           tf.bool, tf.int32, tf.string], list),
       structure.Struct([
           (None, native_platform.AwaitableValueReference(
               _coro(True), computation_types.TensorType(tf.bool))),
           (None, native_platform.AwaitableValueReference(
               _coro(1), computation_types.TensorType(tf.int32))),
           (None, native_platform.AwaitableValueReference(
               _coro('a'), computation_types.TensorType(tf.string))),
       ])),
      ('struct_named',
       _coro(collections.OrderedDict([('a', True), ('b', 1), ('c', 'a')])),
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string),
       ], collections.OrderedDict),
       structure.Struct([
           ('a', native_platform.AwaitableValueReference(
               _coro(True), computation_types.TensorType(tf.bool))),
           ('b', native_platform.AwaitableValueReference(
               _coro(1), computation_types.TensorType(tf.int32))),
           ('c', native_platform.AwaitableValueReference(
               _coro('a'), computation_types.TensorType(tf.string))),
       ])),
      ('struct_nested',
       _coro(collections.OrderedDict([
           ('x', collections.OrderedDict([('a', True), ('b', 1)])),
           ('y', collections.OrderedDict([('c', 'a')])),
       ])),
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', tf.bool),
               ('b', tf.int32),
           ], collections.OrderedDict)),
           ('y', computation_types.StructWithPythonType([
               ('c', tf.string),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
       structure.Struct([
           ('x', structure.Struct([
               ('a', native_platform.AwaitableValueReference(
                   _coro(True), computation_types.TensorType(tf.bool))),
               ('b', native_platform.AwaitableValueReference(
                   _coro(1), computation_types.TensorType(tf.int32))),
           ])),
           ('y', structure.Struct([
               ('c', native_platform.AwaitableValueReference(
                   _coro('a'), computation_types.TensorType(tf.string))),
           ])),
       ])),
  )
  # pyformat: enable
  async def test_returns_value_materialized_multiple(self, awaitable,
                                                     type_signature,
                                                     expected_value):
    actual_value = native_platform._create_structure_of_awaitable_references(
        awaitable=awaitable, type_signature=type_signature)

    if (type_signature.is_struct() and
        not structure.is_same_structure(actual_value, expected_value)):
      self.fail('Expected the structures to be the same, found '
                f'{actual_value} and {expected_value}')
    actual_flattened = structure.flatten(actual_value)
    actual_materialized = await asyncio.gather(
        *[v.get_value() for v in actual_flattened],
        *[v.get_value() for v in actual_flattened],
        *[v.get_value() for v in actual_flattened])
    expected_flattened = structure.flatten(expected_value)
    expected_materialized = await asyncio.gather(
        *[v.get_value() for v in expected_flattened],
        *[v.get_value() for v in actual_flattened],
        *[v.get_value() for v in actual_flattened])
    self.assertEqual(actual_materialized, expected_materialized)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_awaitable(self, awaitable):
    type_signature = computation_types.TensorType(tf.int32)

    with self.assertRaises(TypeError):
      native_platform._create_structure_of_awaitable_references(
          awaitable=awaitable, type_signature=type_signature)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_type_signature(self, type_signature):
    awaitable = _coro(1)

    with self.assertRaises(TypeError):
      native_platform._create_structure_of_awaitable_references(
          awaitable=awaitable, type_signature=type_signature)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated', computation_types.FederatedType(
          tf.int32, placements.CLIENTS)),
      ('function', computation_types.FunctionType(tf.int32, tf.int32)),
      ('placement', computation_types.PlacementType()),
  )
  # pyformat: enable
  def test_raises_not_implemented_error_with_type_signature(
      self, type_signature):
    awaitable = _coro(1)

    with self.assertRaises(NotImplementedError):
      native_platform._create_structure_of_awaitable_references(
          awaitable=awaitable, type_signature=type_signature)


class MaterializeStructureOfValueReferencesTest(parameterized.TestCase,
                                                unittest.IsolatedAsyncioTestCase
                                               ):

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor',
       native_platform.AwaitableValueReference(
           _coro(1), computation_types.TensorType(tf.int32)),
       computation_types.TensorType(tf.int32),
       1),
      ('federated',
       native_platform.AwaitableValueReference(
           _coro(1), computation_types.TensorType(tf.int32)),
       computation_types.FederatedType(tf.int32, placements.SERVER),
       1),
      ('struct_unnamed',
       [
           native_platform.AwaitableValueReference(
               _coro(True), computation_types.TensorType(tf.bool)),
           native_platform.AwaitableValueReference(
               _coro(1), computation_types.TensorType(tf.int32)),
           native_platform.AwaitableValueReference(
               _coro('a'), computation_types.TensorType(tf.string)),
       ],
       computation_types.StructWithPythonType(
           [tf.bool, tf.int32, tf.string], list),
       structure.Struct([(None, True), (None, 1), (None, 'a')])),
      ('struct_named',
       collections.OrderedDict([
           ('a', native_platform.AwaitableValueReference(
               _coro(True), computation_types.TensorType(tf.bool))),
           ('b', native_platform.AwaitableValueReference(
               _coro(1), computation_types.TensorType(tf.int32))),
           ('c', native_platform.AwaitableValueReference(
               _coro('a'), computation_types.TensorType(tf.string))),
       ]),
       computation_types.StructWithPythonType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string)
       ], collections.OrderedDict),
       structure.Struct([('a', True), ('b', 1), ('c', 'a')])),
      ('struct_nested',
       collections.OrderedDict([
           ('x', collections.OrderedDict([
               ('a', native_platform.AwaitableValueReference(
                   _coro(True), computation_types.TensorType(tf.bool))),
               ('b', native_platform.AwaitableValueReference(
                   _coro(1), computation_types.TensorType(tf.int32))),
           ])),
           ('y', collections.OrderedDict([
               ('c', native_platform.AwaitableValueReference(
                   _coro('a'), computation_types.TensorType(tf.string))),
           ])),
       ]),
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', tf.bool),
               ('b', tf.int32),
           ], collections.OrderedDict)),
           ('y', computation_types.StructWithPythonType([
               ('c', tf.string),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
       structure.Struct([
           ('x', structure.Struct([
               ('a', True),
               ('b', 1),
           ])),
           ('y', structure.Struct([
               ('c', 'a'),
           ])),
       ])),
  )
  # pyformat: enable
  async def test_returns_value(self, value, type_signature, expected_value):
    actual_value = await native_platform._materialize_structure_of_value_references(
        value=value, type_signature=type_signature)

    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_type_signature(self, type_signature):
    with self.assertRaises(TypeError):
      await native_platform._materialize_structure_of_value_references(
          value=1, type_signature=type_signature)


class NativeFederatedContextTest(parameterized.TestCase,
                                 unittest.IsolatedAsyncioTestCase,
                                 tf.test.TestCase):

  def test_init_does_not_raise_type_error_with_context(self):
    context = execution_contexts.create_local_async_python_execution_context()

    try:
      native_platform.NativeFederatedContext(context)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_init_raises_type_error_with_context(self, context):
    with self.assertRaises(TypeError):
      native_platform.NativeFederatedContext(context)

  def test_init_raises_value_error_with_context(self):
    context = execution_contexts.create_local_python_execution_context()

    with self.assertRaises(ValueError):
      native_platform.NativeFederatedContext(context)

  async def test_invoke_returns_result(self):
    context = execution_contexts.create_local_async_python_execution_context()
    context = native_platform.NativeFederatedContext(context)

    @tensorflow_computation.tf_computation(tf.int32, tf.int32)
    def add(x, y):
      return x + y

    result = context.invoke(add, structure.Struct.unnamed(1, 2))
    actual_value = await result.get_value()
    self.assertEqual(actual_value, 3)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_invoke_raises_type_error_with_comp(self, comp):
    context = execution_contexts.create_local_async_python_execution_context()
    context = native_platform.NativeFederatedContext(context)

    with self.assertRaises(TypeError):
      context.invoke(comp, None)

  def test_invoke_raises_value_error_with_comp(self):
    context = execution_contexts.create_local_async_python_execution_context()
    context = native_platform.NativeFederatedContext(context)

    @tensorflow_computation.tf_computation()
    def return_one():
      return 1

    with self.assertRaises(ValueError):
      with mock.patch.object(
          federated_context,
          'contains_only_server_placed_data',
          return_value=False):
        context.invoke(return_one, None)

  async def test_call_computation_returns_result(self):
    context = execution_contexts.create_local_async_python_execution_context()
    context = native_platform.NativeFederatedContext(context)

    @tensorflow_computation.tf_computation(tf.int32, tf.int32)
    def add(x, y):
      return x + y

    with context_stack_impl.context_stack.install(context):
      result = add(1, 2)

    actual_value = await result.get_value()
    self.assertEqual(actual_value, 3)


class DatasetDataSourceIteratorTest(parameterized.TestCase, tf.test.TestCase):

  def test_init_does_not_raise_type_error(self):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(tf.int32), placements.CLIENTS)

    try:
      native_platform.DatasetDataSourceIterator(
          datasets=datasets, federated_type=federated_type)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_datasets(self, datasets):
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(tf.int32), placements.CLIENTS)

    with self.assertRaises(TypeError):
      native_platform.DatasetDataSourceIterator(
          datasets=datasets, federated_type=federated_type)

  # pyformat: disable
  @parameterized.named_parameters(
      ('function', computation_types.FunctionType(tf.int32, tf.int32)),
      ('placement', computation_types.PlacementType()),
      ('sequence', computation_types.SequenceType(tf.int32)),
      ('struct', computation_types.StructWithPythonType(
          [tf.bool, tf.int32, tf.string], list)),
      ('tensor', computation_types.TensorType(tf.int32)),
  )
  # pyformat: enable
  def test_init_raises_type_error_with_federated_type(self, federated_type):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3

    with self.assertRaises(TypeError):
      native_platform.DatasetDataSourceIterator(
          datasets=datasets, federated_type=federated_type)

  def test_init_raises_value_error_with_datasets_empty(self):
    datasets = []
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(tf.int32), placements.CLIENTS)

    with self.assertRaises(ValueError):
      native_platform.DatasetDataSourceIterator(
          datasets=datasets, federated_type=federated_type)

  def test_init_raises_value_error_with_datasets_different_types(self):
    datasets = [
        tf.data.Dataset.from_tensor_slices([1, 2, 3]),
        tf.data.Dataset.from_tensor_slices(['a', 'b', 'c']),
    ]
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(tf.int32), placements.CLIENTS)

    with self.assertRaises(ValueError):
      native_platform.DatasetDataSourceIterator(
          datasets=datasets, federated_type=federated_type)

  @parameterized.named_parameters(
      ('1', 0),
      ('2', 1),
      ('3', 2),
  )
  def test_select_returns_data(self, number_of_clients):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(tf.int32), placements.CLIENTS)
    iterator = native_platform.DatasetDataSourceIterator(
        datasets=datasets, federated_type=federated_type)

    data = iterator.select(number_of_clients)

    self.assertLen(data, number_of_clients)
    for actual_dataset in data:
      expected_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
      self.assertSameElements(actual_dataset, expected_dataset)

  @parameterized.named_parameters(
      ('str', 'a'),
      ('list', []),
  )
  def test_select_raises_type_error_with_number_of_clients(
      self, number_of_clients):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(tf.int32), placements.CLIENTS)
    iterator = native_platform.DatasetDataSourceIterator(
        datasets=datasets, federated_type=federated_type)

    with self.assertRaises(TypeError):
      iterator.select(number_of_clients)

  @parameterized.named_parameters(
      ('none', None),
      ('negative', -1),
      ('greater', 4),
  )
  def test_select_raises_value_error_with_number_of_clients(
      self, number_of_clients):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(tf.int32), placements.CLIENTS)
    iterator = native_platform.DatasetDataSourceIterator(
        datasets=datasets, federated_type=federated_type)

    with self.assertRaises(ValueError):
      iterator.select(number_of_clients)


class DatasetDataSourceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', [1, 2, 3], tf.int32),
      ('str', ['a', 'b', 'c'], tf.string),
  )
  def test_init_sets_federated_type(self, tensors, dtype):
    datasets = [tf.data.Dataset.from_tensor_slices(tensors)] * 3

    data_source = native_platform.DatasetDataSource(datasets)

    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(dtype), placements.CLIENTS)
    self.assertEqual(data_source.federated_type, federated_type)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_datasets(self, datasets):
    with self.assertRaises(TypeError):
      native_platform.DatasetDataSource(datasets)

  def test_init_raises_value_error_with_datasets_empty(self):
    datasets = []

    with self.assertRaises(ValueError):
      native_platform.DatasetDataSource(datasets)

  def test_init_raises_value_error_with_datasets_different_types(self):
    datasets = [
        tf.data.Dataset.from_tensor_slices([1, 2, 3]),
        tf.data.Dataset.from_tensor_slices(['a', 'b', 'c']),
    ]

    with self.assertRaises(ValueError):
      native_platform.DatasetDataSource(datasets)


class ClientIdDataSourceIteratorTest(parameterized.TestCase, tf.test.TestCase):

  def test_init_does_not_raise_type_error(self):
    client_ids = ['a', 'b', 'c']
    try:
      native_platform.ClientIdDataSourceIterator(client_ids=client_ids)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_client_ids(self, client_ids):
    with self.assertRaises(TypeError):
      native_platform.ClientIdDataSourceIterator(client_ids=client_ids)

  def test_init_raises_value_error_with_client_ids_empty(self):
    client_ids = []
    with self.assertRaises(ValueError):
      native_platform.ClientIdDataSourceIterator(client_ids=client_ids)

  @parameterized.named_parameters(
      ('1', 0),
      ('2', 1),
      ('3', 2),
  )
  def test_select_returns_client_ids(self, number_of_clients):
    client_ids = ['a', 'b', 'c']
    iterator = native_platform.ClientIdDataSourceIterator(client_ids=client_ids)

    selected_client_ids = iterator.select(number_of_clients)

    self.assertLen(selected_client_ids, number_of_clients)
    self.assertAllInSet(selected_client_ids, client_ids)
    for client_id in selected_client_ids:
      self.assertDTypeEqual(client_id, tf.string)

  @parameterized.named_parameters(
      ('str', 'a'),
      ('list', []),
  )
  def test_select_raises_type_error_with_number_of_clients(
      self, number_of_clients):
    client_ids = ['a', 'b', 'c']
    iterator = native_platform.ClientIdDataSourceIterator(client_ids=client_ids)

    with self.assertRaises(TypeError):
      iterator.select(number_of_clients)

  @parameterized.named_parameters(
      ('none', None),
      ('negative', -1),
      ('greater', 4),
  )
  def test_select_raises_value_error_with_number_of_clients(
      self, number_of_clients):
    client_ids = ['a', 'b', 'c']
    iterator = native_platform.ClientIdDataSourceIterator(client_ids=client_ids)

    with self.assertRaises(ValueError):
      iterator.select(number_of_clients)


class ClientIdDataSourceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_client_ids(self, client_ids):
    with self.assertRaises(TypeError):
      native_platform.ClientIdDataSource(client_ids)

  def test_init_raises_value_error_with_client_ids_empty(self):
    client_ids = []
    with self.assertRaises(ValueError):
      native_platform.ClientIdDataSource(client_ids)


if __name__ == '__main__':
  absltest.main()
