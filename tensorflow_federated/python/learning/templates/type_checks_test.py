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

import tensorflow as tf

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning.templates import type_checks


class TypeChecksTest(tf.test.TestCase):

  def test_does_not_raise_on_client_placed_sequence(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    type_spec = computation_types.at_clients(sequence_type)
    type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_does_not_raise_on_client_placed_struct_of_sequences(self):
    sequence_type1 = computation_types.SequenceType(tf.int32)
    sequence_type2 = computation_types.SequenceType(tf.float32)
    struct_type = computation_types.StructWithPythonType(
        [sequence_type1, sequence_type2], list
    )
    type_spec = computation_types.at_clients(struct_type)
    type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_server_placed_sequence(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    type_spec = computation_types.at_server(sequence_type)
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_server_placed_struct_of_sequences(self):
    sequence_type1 = computation_types.SequenceType(tf.int32)
    sequence_type2 = computation_types.SequenceType(tf.float32)
    struct_type = computation_types.StructWithPythonType(
        [sequence_type1, sequence_type2], list
    )
    type_spec = computation_types.at_server(struct_type)
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_client_placed_tensor(self):
    tensor_spec = computation_types.TensorType(dtype=tf.int32, shape=(1, 2))
    type_spec = computation_types.at_clients(tensor_spec)
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_client_placed_structure_of_tensor_and_sequence(self):
    tensor_spec = computation_types.TensorType(dtype=tf.int32, shape=(1, 2))
    sequence_type = computation_types.SequenceType(tf.int32)
    struct_type = computation_types.StructWithPythonType(
        [tensor_spec, sequence_type], list
    )
    type_spec = computation_types.at_clients(struct_type)
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_structure_of_client_placed_sequences(self):
    clients_sequence_type = computation_types.at_clients(
        computation_types.SequenceType(tf.int32)
    )
    type_spec = computation_types.StructType([
        (None, clients_sequence_type),
        (None, clients_sequence_type),
    ])
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)


if __name__ == "__main__":
  tf.test.main()
