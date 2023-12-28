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
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.common_libs import async_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import native_platform
from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference


def _value_fn_factory(value: object) -> object:
  async def _value_fn() -> object:
    return value

  return _value_fn


def _identity_federated_computation_factory(
    type_signature: computation_types.Type,
) -> computation_base.Computation:
  @federated_computation.federated_computation(type_signature)
  def _identity(value: object) -> object:
    return value

  return _identity


def _identity_tensorflow_computation_factory(
    type_signature: computation_types.Type,
) -> computation_base.Computation:
  @tensorflow_computation.tf_computation(type_signature)
  def _identity(value: object) -> object:
    return value

  return _identity


class AwaitableValueReferenceTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor',
       _value_fn_factory(1),
       computation_types.TensorType(np.int32)),
      ('sequence',
       _value_fn_factory([1, 2, 3]),
       computation_types.SequenceType(np.int32)),
  )
  # pyformat: enable
  def test_init_does_not_raise_type_error(self, fn, type_signature):
    try:
      native_platform.AwaitableValueReference(fn, type_signature)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_init_raises_type_error_with_fn(self, fn):
    type_signature = computation_types.TensorType(np.int32)

    with self.assertRaises(TypeError):
      native_platform.AwaitableValueReference(fn, type_signature)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated',
       computation_types.FederatedType(np.int32, placements.SERVER)),
      ('struct', computation_types.StructWithPythonType([], list)),
  )
  # pyformat: enable
  def test_init_raises_type_error_with_type_signature(self, type_signature):
    fn = _value_fn_factory(1)

    with self.assertRaises(TypeError):
      native_platform.AwaitableValueReference(fn, type_signature)

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor_bool',
       _value_fn_factory(True),
       computation_types.TensorType(np.bool_),
       True),
      ('tensor_int',
       _value_fn_factory(1),
       computation_types.TensorType(np.int32),
       1),
      ('tensor_str',
       _value_fn_factory('a'),
       computation_types.TensorType(np.str_),
       'a'),
      ('sequence',
       _value_fn_factory(tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       computation_types.SequenceType(np.int32),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),
  )
  # pyformat: enable
  async def test_get_value_returns_value(
      self, fn, type_signature, expected_value
  ):
    reference = native_platform.AwaitableValueReference(fn, type_signature)

    actual_value = await reference.get_value()

    tree.assert_same_structure(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)


class WrapInSharedAwaitableTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_returns_value_shared(self):
    async def _identity(value):
      return value

    fn = native_platform._wrap_in_shared_awaitable(_identity)
    awaitable = fn(1)
    value_1 = await awaitable
    value_2 = await awaitable
    self.assertIs(value_1, value_2)

  def test_returns_value_cached(self):
    async def _identity(value):
      return value

    fn = native_platform._wrap_in_shared_awaitable(_identity)

    awaitable_1 = fn(1)
    awaitable_2 = fn(1)
    self.assertIs(awaitable_1, awaitable_2)
    awaitable_3 = fn(2)
    self.assertIsInstance(awaitable_1, async_utils.SharedAwaitable)
    self.assertIsInstance(awaitable_3, async_utils.SharedAwaitable)
    self.assertIsNot(awaitable_1, awaitable_3)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_fn(self, fn):
    with self.assertRaises(TypeError):
      native_platform._wrap_in_shared_awaitable(fn)


class CreateStructureOfAwaitableReferencesTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor',
       _value_fn_factory(1),
       computation_types.TensorType(np.int32),
       native_platform.AwaitableValueReference(
           _value_fn_factory(1), computation_types.TensorType(np.int32))),
      ('sequence',
       _value_fn_factory(tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       computation_types.SequenceType(np.int32),
       native_platform.AwaitableValueReference(
           _value_fn_factory(tf.data.Dataset.from_tensor_slices([1, 2, 3])),
           computation_types.SequenceType(np.int32))),
      ('federated_server',
       _value_fn_factory(1),
       computation_types.FederatedType(np.int32, placements.SERVER),
       native_platform.AwaitableValueReference(
           _value_fn_factory(1), computation_types.TensorType(np.int32))),
      ('struct_unnamed',
       _value_fn_factory([True, 1, 'a']),
       computation_types.StructWithPythonType([
           np.bool_, np.int32, np.str_], list),
       structure.Struct([
           (None, native_platform.AwaitableValueReference(
               _value_fn_factory(True), computation_types.TensorType(np.bool_))),
           (None, native_platform.AwaitableValueReference(
               _value_fn_factory(1), computation_types.TensorType(np.int32))),
           (None, native_platform.AwaitableValueReference(
               _value_fn_factory('a'), computation_types.TensorType(np.str_))),
       ])),
      ('struct_named',
       _value_fn_factory({'a': True, 'b': 1, 'c': 'a'}),
       computation_types.StructWithPythonType([
           ('a', np.bool_),
           ('b', np.int32),
           ('c', np.str_),
       ], collections.OrderedDict),
       structure.Struct([
           ('a', native_platform.AwaitableValueReference(
               _value_fn_factory(True), computation_types.TensorType(np.bool_))),
           ('b', native_platform.AwaitableValueReference(
               _value_fn_factory(1), computation_types.TensorType(np.int32))),
           ('c', native_platform.AwaitableValueReference(
               _value_fn_factory('a'), computation_types.TensorType(np.str_))),
       ])),
      ('struct_nested',
       _value_fn_factory({'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}}),
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', np.bool_),
               ('b', np.int32),
           ], collections.OrderedDict)),
           ('y', computation_types.StructWithPythonType([
               ('c', np.str_),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
       structure.Struct([
           ('x', structure.Struct([
               ('a', native_platform.AwaitableValueReference(
                   _value_fn_factory(True), computation_types.TensorType(np.bool_))),
               ('b', native_platform.AwaitableValueReference(
                   _value_fn_factory(1), computation_types.TensorType(np.int32))),
           ])),
           ('y', structure.Struct([
               ('c', native_platform.AwaitableValueReference(
                   _value_fn_factory('a'), computation_types.TensorType(np.str_))),
           ])),
       ])),
  )
  # pyformat: enable
  async def test_returns_value(self, fn, type_signature, expected_value):
    actual_value = native_platform._create_structure_of_awaitable_references(
        fn, type_signature
    )

    if isinstance(actual_value, structure.Struct) and isinstance(
        expected_value, structure.Struct
    ):
      actual_value = structure.flatten(actual_value)
      expected_value = structure.flatten(expected_value)
    actual_value = await value_reference.materialize_value(actual_value)
    expected_value = await value_reference.materialize_value(expected_value)
    tree.assert_same_structure(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_fn(self, fn):
    type_signature = computation_types.TensorType(np.int32)

    with self.assertRaises(TypeError):
      native_platform._create_structure_of_awaitable_references(
          fn, type_signature
      )

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_type_signature(self, type_signature):
    fn = _value_fn_factory(1)

    with self.assertRaises(TypeError):
      native_platform._create_structure_of_awaitable_references(
          fn, type_signature
      )

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_clients', computation_types.FederatedType(
          np.int32, placements.CLIENTS)),
      ('function', computation_types.FunctionType(np.int32, np.int32)),
      ('placement', computation_types.PlacementType()),
  )
  # pyformat: enable
  def test_raises_not_implemented_error_with_type_signature(
      self, type_signature
  ):
    fn = _value_fn_factory(1)

    with self.assertRaises(NotImplementedError):
      native_platform._create_structure_of_awaitable_references(
          fn, type_signature
      )


class MaterializeStructureOfValueReferencesTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor',
       native_platform.AwaitableValueReference(
           _value_fn_factory(1), computation_types.TensorType(np.int32)),
       computation_types.TensorType(np.int32),
       1),
      ('sequence',
       native_platform.AwaitableValueReference(
           _value_fn_factory(tf.data.Dataset.from_tensor_slices([1, 2, 3])),
           computation_types.SequenceType(np.int32)),
       computation_types.SequenceType(np.int32),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),
      ('federated_server',
       native_platform.AwaitableValueReference(
           _value_fn_factory(1), computation_types.TensorType(np.int32)),
       computation_types.FederatedType(np.int32, placements.SERVER),
       1),
      ('federated_clients',
       native_platform.AwaitableValueReference(
           _value_fn_factory(1), computation_types.TensorType(np.int32)),
       computation_types.FederatedType(np.int32, placements.CLIENTS),
       1),
      ('struct_unnamed',
       [
           native_platform.AwaitableValueReference(
               _value_fn_factory(True), computation_types.TensorType(np.bool_)),
           native_platform.AwaitableValueReference(
               _value_fn_factory(1), computation_types.TensorType(np.int32)),
           native_platform.AwaitableValueReference(
               _value_fn_factory('a'), computation_types.TensorType(np.str_)),
       ],
       computation_types.StructWithPythonType(
           [np.bool_, np.int32, np.str_], list),
       structure.Struct([(None, True), (None, 1), (None, 'a')])),
      ('struct_named',
       {
           'a': native_platform.AwaitableValueReference(
               _value_fn_factory(True), computation_types.TensorType(np.bool_)),
           'b': native_platform.AwaitableValueReference(
               _value_fn_factory(1), computation_types.TensorType(np.int32)),
           'c': native_platform.AwaitableValueReference(
               _value_fn_factory('a'), computation_types.TensorType(np.str_)),
       },
       computation_types.StructWithPythonType([
           ('a', np.bool_),
           ('b', np.int32),
           ('c', np.str_)
       ], collections.OrderedDict),
       structure.Struct([('a', True), ('b', 1), ('c', 'a')])),
      ('struct_nested',
       {
           'x': {
               'a': native_platform.AwaitableValueReference(
                   _value_fn_factory(True),
                   computation_types.TensorType(np.bool_)),
               'b': native_platform.AwaitableValueReference(
                   _value_fn_factory(1),
                   computation_types.TensorType(np.int32)),
           },
           'y': {
               'c': native_platform.AwaitableValueReference(
                   _value_fn_factory('a'),
                   computation_types.TensorType(np.str_)),
           }
       },
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', np.bool_),
               ('b', np.int32),
           ], collections.OrderedDict)),
           ('y', computation_types.StructWithPythonType([
               ('c', np.str_),
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
    actual_value = (
        await native_platform._materialize_structure_of_value_references(
            value, type_signature
        )
    )

    tree.assert_same_structure(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_type_signature(self, type_signature):
    value = 1

    with self.assertRaises(TypeError):
      await native_platform._materialize_structure_of_value_references(
          value, type_signature
      )

  # pyformat: disable
  @parameterized.named_parameters(
      ('function', computation_types.FunctionType(np.int32, np.int32)),
      ('placement', computation_types.PlacementType()),
  )
  # pyformat: enable
  async def test_raises_not_implemented_error_with_type_signature(
      self, type_signature
  ):
    value = 1

    with self.assertRaises(NotImplementedError):
      await native_platform._materialize_structure_of_value_references(
          value, type_signature
      )


class NativeFederatedContextTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def test_init_does_not_raise_type_error(self):
    context = execution_contexts.create_async_local_cpp_execution_context()
    try:
      native_platform.NativeFederatedContext(context)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  # pyformat: disable
  @parameterized.named_parameters(
      ('sync_cpp',
       execution_contexts.create_sync_local_cpp_execution_context()),
  )
  # pyformat: enable
  def test_init_raises_type_error_with_context(self, context):
    with self.assertRaises(TypeError):
      native_platform.NativeFederatedContext(context)

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor',
       computation_types.TensorType(np.int32),
       1),
      ('sequence',
       computation_types.SequenceType(np.int32),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),
      ('federated_server',
       computation_types.FederatedType(np.int32, placements.SERVER),
       1),
      ('federated_clients',
       computation_types.FederatedType(
           np.int32, placements.CLIENTS, all_equal=True),
       1),
      ('struct_unnamed',
       computation_types.StructWithPythonType(
           [np.bool_, np.int32, np.str_], list),
       [True, 1, 'a']),
      ('struct_named',
       computation_types.StructWithPythonType([
           ('a', np.bool_),
           ('b', np.int32),
           ('c', np.str_)
       ], collections.OrderedDict),
       {'a': True, 'b': 1, 'c': 'a'}),
      ('struct_nested',
       computation_types.StructWithPythonType([
           ('x', computation_types.StructWithPythonType([
               ('a', np.bool_),
               ('b', np.int32),
           ], collections.OrderedDict)),
           ('y', computation_types.StructWithPythonType([
               ('c', np.str_),
           ], collections.OrderedDict)),
       ], collections.OrderedDict),
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}}),
  )
  # pyformat: enable
  async def test_invoke_accepts_argument(self, type_signature, arg):
    context = execution_contexts.create_async_local_cpp_execution_context()
    context = native_platform.NativeFederatedContext(context)

    @federated_computation.federated_computation(type_signature)
    def _comp(value):
      del value  # Unused.
      return 1

    try:
      with program_test_utils.assert_not_warns(RuntimeWarning):
        result = context.invoke(_comp, arg)
        _ = await result.get_value()
    except NotImplementedError:
      self.fail('Raised `NotImplementedError` unexpectedly.')

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor',
       _identity_federated_computation_factory(
           computation_types.TensorType(np.int32)),
       1,
       1),
      ('sequence',
       _identity_tensorflow_computation_factory(
           computation_types.SequenceType(np.int32)),
       tf.data.Dataset.from_tensor_slices([1, 2, 3]),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),
      ('federated_server',
       _identity_federated_computation_factory(
           computation_types.FederatedType(np.int32, placements.SERVER)),
       1,
       1),
      ('struct_unnamed',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType(
               [np.bool_, np.int32, np.str_], list)),
       [True, 1, 'a'],
       [True, 1, b'a']),
      ('struct_named',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([
               ('a', np.bool_),
               ('b', np.int32),
               ('c', np.str_)
           ], collections.OrderedDict)),
       {'a': True, 'b': 1, 'c': 'a'},
       {'a': True, 'b': 1, 'c': b'a'}),
  )
  # pyformat: enable
  async def test_invoke_returns_result(self, comp, arg, expected_value):
    context = execution_contexts.create_async_local_cpp_execution_context()
    context = native_platform.NativeFederatedContext(context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await value_reference.materialize_value(result)

    tree.assert_same_structure(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)

  # pyformat: disable
  @parameterized.named_parameters(
      ('struct_nested',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([
               ('x', computation_types.StructWithPythonType([
                   ('a', np.bool_),
                   ('b', np.int32),
               ], collections.OrderedDict)),
               ('y', computation_types.StructWithPythonType([
                   ('c', np.str_),
               ], collections.OrderedDict)),
           ], collections.OrderedDict)),
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       {'x': {'a': True, 'b': 1}, 'y': {'c': b'a'}}),
      ('struct_partially_empty',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([
               ('x', computation_types.StructWithPythonType([
                   ('a', np.bool_),
                   ('b', np.int32),
               ], collections.OrderedDict)),
               ('y', computation_types.StructWithPythonType(
                   [], collections.OrderedDict)),
           ], collections.OrderedDict)),
       {'x': {'a': True, 'b': 1}, 'y': {}},
       {'x': {'a': True, 'b': 1}, 'y': {}}),
  )
  # pyformat: enable
  async def test_invoke_returns_result_materialized_sequentially(
      self, comp, arg, expected_value
  ):
    context = execution_contexts.create_async_local_cpp_execution_context()
    mock_context = mock.Mock(
        spec=async_execution_context.AsyncExecutionContext, wraps=context
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      flattened = structure_utils.flatten(result)
      materialized = [await v.get_value() for v in flattened]
      actual_value = structure_utils.unflatten_as(result, materialized)

    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once()

  # pyformat: disable
  @parameterized.named_parameters(
      ('struct_nested',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([
               ('x', computation_types.StructWithPythonType([
                   ('a', np.bool_),
                   ('b', np.int32),
               ], collections.OrderedDict)),
               ('y', computation_types.StructWithPythonType([
                   ('c', np.str_),
               ], collections.OrderedDict)),
           ], collections.OrderedDict)),
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       {'x': {'a': True, 'b': 1}, 'y': {'c': b'a'}}),
      ('struct_partially_empty',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([
               ('x', computation_types.StructWithPythonType([
                   ('a', np.bool_),
                   ('b', np.int32),
               ], collections.OrderedDict)),
               ('y', computation_types.StructWithPythonType(
                   [], collections.OrderedDict)),
           ], collections.OrderedDict)),
       {'x': {'a': True, 'b': 1}, 'y': {}},
       {'x': {'a': True, 'b': 1}, 'y': {}}),
  )
  # pyformat: enable
  async def test_invoke_returns_result_materialized_concurrently(
      self, comp, arg, expected_value
  ):
    context = execution_contexts.create_async_local_cpp_execution_context()
    mock_context = mock.Mock(
        spec=async_execution_context.AsyncExecutionContext, wraps=context
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await value_reference.materialize_value(result)

    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once()

  # pyformat: disable
  @parameterized.named_parameters(
      ('struct_nested',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([
               ('x', computation_types.StructWithPythonType([
                   ('a', np.bool_),
                   ('b', np.int32),
               ], collections.OrderedDict)),
               ('y', computation_types.StructWithPythonType([
                   ('c', np.str_),
               ], collections.OrderedDict)),
           ], collections.OrderedDict)),
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       {'x': {'a': True, 'b': 1}, 'y': {'c': b'a'}}),
      ('struct_partially_empty',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([
               ('x', computation_types.StructWithPythonType([
                   ('a', np.bool_),
                   ('b', np.int32),
               ], collections.OrderedDict)),
               ('y', computation_types.StructWithPythonType(
                   [], collections.OrderedDict)),
           ], collections.OrderedDict)),
       {'x': {'a': True, 'b': 1}, 'y': {}},
       {'x': {'a': True, 'b': 1}, 'y': {}}),
  )
  # pyformat: enable
  async def test_invoke_returns_result_materialized_multiple(
      self, comp, arg, expected_value
  ):
    context = execution_contexts.create_async_local_cpp_execution_context()
    mock_context = mock.Mock(
        spec=async_execution_context.AsyncExecutionContext, wraps=context
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await asyncio.gather(
          value_reference.materialize_value(result),
          value_reference.materialize_value(result),
          value_reference.materialize_value(result),
      )

    expected_value = [expected_value] * 3
    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once()

  # pyformat: disable
  @parameterized.named_parameters(
      ('struct_unnamed_empty',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([], list)),
       [],
       []),
      ('struct_named_empty',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([], collections.OrderedDict)),
       {},
       {}),
      ('struct_nested_empty',
       _identity_federated_computation_factory(
           computation_types.StructWithPythonType([
               ('x', computation_types.StructWithPythonType(
                   [], collections.OrderedDict)),
               ('y', computation_types.StructWithPythonType(
                   [], collections.OrderedDict)),
           ], collections.OrderedDict)),
       {'x': {}, 'y': {}},
       {'x': {}, 'y': {}}),
  )
  # pyformat: enable
  async def test_invoke_returns_result_comp_not_called(
      self, comp, arg, expected_value
  ):
    context = execution_contexts.create_async_local_cpp_execution_context()
    mock_context = mock.Mock(
        spec=async_execution_context.AsyncExecutionContext, wraps=context
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await value_reference.materialize_value(result)

    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_not_called()

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_invoke_raises_type_error_with_comp(self, comp):
    context = execution_contexts.create_async_local_cpp_execution_context()
    context = native_platform.NativeFederatedContext(context)

    with self.assertRaises(TypeError):
      context.invoke(comp, None)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_clients',
       _identity_federated_computation_factory(
           computation_types.FederatedType(np.int32, placements.CLIENTS)),
       1),
      ('function',
       _identity_federated_computation_factory(
           computation_types.FunctionType(np.int32, np.int32)),
       _identity_federated_computation_factory(
           computation_types.TensorType(np.int32))),
      ('placement',
       _identity_federated_computation_factory(
           computation_types.PlacementType()),
       None),
  )
  # pyformat: enable
  def test_invoke_raises_value_error_with_comp(self, comp, arg):
    context = execution_contexts.create_async_local_cpp_execution_context()
    context = native_platform.NativeFederatedContext(context)

    with self.assertRaises(ValueError):
      context.invoke(comp, arg)


if __name__ == '__main__':
  absltest.main()
