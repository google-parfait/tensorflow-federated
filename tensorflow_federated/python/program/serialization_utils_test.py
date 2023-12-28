# Copyright 2023, The TensorFlow Federated Authors.
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

import struct
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import serialization_utils


class _TestNamedTuple(NamedTuple):
  a: object
  b: object
  c: object


class SerializationUtilsStrTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty', ''),
      ('short', 'abc'),
      ('long', 'abc' * 100),
  )
  def test_pack_and_unpack_str(self, value):
    value_bytes = serialization_utils.pack_str(value)
    actual_value, actual_size = serialization_utils.unpack_str_from(value_bytes)

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_str_with_offset(self):
    value = 'abc'
    offset = 100

    value_bytes = serialization_utils.pack_str(value)
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_str, actual_size = serialization_utils.unpack_str_from(
        padded_bytes, offset
    )

    self.assertEqual(actual_str, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_str_from_raises_struct_error_with_offset(self, offset):
    value = 'abc'
    value_bytes = serialization_utils.pack_str(value)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_str_from(value_bytes, offset)

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_str_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = 'abc'
    value_bytes = serialization_utils.pack_str(value)
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_str_from(corrupt_bytes)


class SerializationUtilsSequenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty', []),
      ('short', ['abc', 'def', 'ghi']),
      ('long', ['abc', 'def', 'ghi'] * 100),
  )
  def test_pack_and_unpack_sequence(self, value):
    value_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_str, value
    )
    actual_value, actual_size = serialization_utils.unpack_sequence_from(
        serialization_utils.unpack_str_from, value_bytes
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_sequence_offset(self):
    value = ['abc', 'def', 'ghi']
    offset = 100

    value_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_str, value
    )
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_value, actual_size = serialization_utils.unpack_sequence_from(
        serialization_utils.unpack_str_from, padded_bytes, offset
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_sequence_from_raises_struct_error_with_offset(self, offset):
    value = ['abc', 'def', 'ghi']
    value_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_str, value
    )

    with self.assertRaises(struct.error):
      serialization_utils.unpack_sequence_from(
          serialization_utils.unpack_str_from, value_bytes, offset
      )

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_sequence_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = ['abc', 'def', 'ghi']
    value_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_str, value
    )
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_sequence_from(
          serialization_utils.unpack_str_from, corrupt_bytes
      )


class SerializationUtilsSerializableTest(parameterized.TestCase):

  def test_pack_and_unpack_serializable(self):
    value = program_test_utils.TestSerializable(1, 2)

    value_bytes = serialization_utils.pack_serializable(value)
    actual_value, actual_size = serialization_utils.unpack_serializable_from(
        value_bytes
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_serializable_with_offset(self):
    value = program_test_utils.TestSerializable(1, 2)
    offset = 100

    value_bytes = serialization_utils.pack_serializable(value)
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_value, actual_size = serialization_utils.unpack_serializable_from(
        padded_bytes, offset
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_serializable_from_raises_struct_error_with_offset(
      self, offset
  ):
    value = program_test_utils.TestSerializable(1, 2)
    value_bytes = serialization_utils.pack_serializable(value)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_serializable_from(value_bytes, offset)

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_serializable_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = program_test_utils.TestSerializable(1, 2)
    value_bytes = serialization_utils.pack_serializable(value)
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_serializable_from(corrupt_bytes)


class SerializationUtilsTypeSpecTest(parameterized.TestCase):

  def test_pack_and_unpack_type_spec(self):
    value = computation_types.TensorType(np.int32)

    value_bytes = serialization_utils.pack_type_spec(value)
    actual_value, actual_size = serialization_utils.unpack_type_spec_from(
        value_bytes,
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_type_spec_with_offset(self):
    value = computation_types.TensorType(np.int32)
    offset = 100

    value_bytes = serialization_utils.pack_type_spec(value)
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_value, actual_size = serialization_utils.unpack_type_spec_from(
        padded_bytes, offset
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_type_spec_from_raises_struct_error_with_offset(self, offset):
    value = computation_types.TensorType(np.int32)
    value_bytes = serialization_utils.pack_type_spec(value)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_type_spec_from(value_bytes, offset)

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_type_spec_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = computation_types.TensorType(np.int32)
    value_bytes = serialization_utils.pack_type_spec(value)
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_type_spec_from(corrupt_bytes)


class SerializationUtilsElementSpecTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('tensor_spec', tf.TensorSpec(shape=(3,), dtype=np.int32, name=None)),
      ('list',
       [
           tf.TensorSpec(shape=(3,), dtype=np.int32, name=None),
           tf.TensorSpec(shape=(3,), dtype=np.int32, name=None),
           tf.TensorSpec(shape=(3,), dtype=np.int32, name=None),
       ]),
      ('dict',
       {
           'a': tf.TensorSpec(shape=(3,), dtype=np.int32, name=None),
           'b': tf.TensorSpec(shape=(3,), dtype=np.int32, name=None),
           'c': tf.TensorSpec(shape=(3,), dtype=np.int32, name=None),
       }),
      ('named_tuple',
       _TestNamedTuple(
           a=tf.TensorSpec(shape=(3,), dtype=np.int32, name=None),
           b=tf.TensorSpec(shape=(3,), dtype=np.int32, name=None),
           c=tf.TensorSpec(shape=(3,), dtype=np.int32, name=None),
       )),
  )
  # pyformat: enable
  def test_pack_and_unpack_element_spec(self, value):
    value_bytes = serialization_utils.pack_element_spec(value)
    actual_value, actual_size = serialization_utils.unpack_element_spec_from(
        value_bytes,
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_element_spec_with_offset(self):
    value = tf.TensorSpec(shape=(3,), dtype=np.int32, name=None)
    offset = 100

    value_bytes = serialization_utils.pack_element_spec(value)
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_value, actual_size = serialization_utils.unpack_element_spec_from(
        padded_bytes, offset
    )

    self.assertEqual(actual_value, value)
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_element_spec_from_raises_struct_error_with_offset(
      self, offset
  ):
    value = tf.TensorSpec(shape=(3,), dtype=np.int32, name=None)
    value_bytes = serialization_utils.pack_element_spec(value)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_element_spec_from(value_bytes, offset)

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_element_spec_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = tf.TensorSpec(shape=(3,), dtype=np.int32, name=None)
    value_bytes = serialization_utils.pack_element_spec(value)
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_element_spec_from(corrupt_bytes)


class SerializationUtilsDatasetTest(parameterized.TestCase):

  def test_pack_and_unpack_dataset(self):
    value = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    value_bytes = serialization_utils.pack_dataset(value)
    actual_value, actual_size = serialization_utils.unpack_dataset_from(
        value_bytes,
    )

    self.assertEqual(list(actual_value), list(value))
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  def test_pack_and_unpack_dataset_with_offset(self):
    value = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    offset = 100

    value_bytes = serialization_utils.pack_dataset(value)
    padded_bytes = bytes([1] * offset) + value_bytes
    actual_value, actual_size = serialization_utils.unpack_dataset_from(
        padded_bytes, offset
    )

    self.assertEqual(list(actual_value), list(value))
    expected_size = len(value_bytes)
    self.assertEqual(actual_size, expected_size)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
  )
  def test_unpack_element_spec_from_raises_struct_error_with_offset(
      self, offset
  ):
    value = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    value_bytes = serialization_utils.pack_dataset(value)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_dataset_from(value_bytes, offset)

  @parameterized.named_parameters(
      ('leading', lambda x: x[1:]),
      ('trailing', lambda x: x[:-1]),
  )
  def test_unpack_dataset_from_raises_struct_error_with_corrupt_bytes(
      self, corrupt_fn
  ):
    value = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    value_bytes = serialization_utils.pack_dataset(value)
    corrupt_bytes = corrupt_fn(value_bytes)

    with self.assertRaises(struct.error):
      serialization_utils.unpack_dataset_from(corrupt_bytes)


if __name__ == '__main__':
  absltest.main()
