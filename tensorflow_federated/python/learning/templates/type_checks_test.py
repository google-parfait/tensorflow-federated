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

import federated_language
import numpy as np
import tensorflow as tf
from tensorflow_federated.python.learning.templates import type_checks


class TypeChecksTest(tf.test.TestCase):

  def test_does_not_raise_on_client_placed_sequence(self):
    sequence_type = federated_language.SequenceType(np.int32)
    type_spec = federated_language.FederatedType(
        sequence_type, federated_language.CLIENTS
    )
    type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_does_not_raise_on_client_placed_struct_of_sequences(self):
    sequence_type1 = federated_language.SequenceType(np.int32)
    sequence_type2 = federated_language.SequenceType(np.float32)
    struct_type = federated_language.StructWithPythonType(
        [sequence_type1, sequence_type2], list
    )
    type_spec = federated_language.FederatedType(
        struct_type, federated_language.CLIENTS
    )
    type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_server_placed_sequence(self):
    sequence_type = federated_language.SequenceType(np.int32)
    type_spec = federated_language.FederatedType(
        sequence_type, federated_language.SERVER
    )
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_server_placed_struct_of_sequences(self):
    sequence_type1 = federated_language.SequenceType(np.int32)
    sequence_type2 = federated_language.SequenceType(np.float32)
    struct_type = federated_language.StructWithPythonType(
        [sequence_type1, sequence_type2], list
    )
    type_spec = federated_language.FederatedType(
        struct_type, federated_language.SERVER
    )
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_client_placed_tensor(self):
    tensor_spec = federated_language.TensorType(np.int32, (1, 2))
    type_spec = federated_language.FederatedType(
        tensor_spec, federated_language.CLIENTS
    )
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_client_placed_structure_of_tensor_and_sequence(self):
    tensor_spec = federated_language.TensorType(np.int32, (1, 2))
    sequence_type = federated_language.SequenceType(np.int32)
    struct_type = federated_language.StructWithPythonType(
        [tensor_spec, sequence_type], list
    )
    type_spec = federated_language.FederatedType(
        struct_type, federated_language.CLIENTS
    )
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)

  def test_raises_on_structure_of_client_placed_sequences(self):
    clients_sequence_type = federated_language.FederatedType(
        federated_language.SequenceType(np.int32), federated_language.CLIENTS
    )
    type_spec = federated_language.StructType([
        (None, clients_sequence_type),
        (None, clients_sequence_type),
    ])
    with self.assertRaises(type_checks.ClientSequenceTypeError):
      type_checks.check_is_client_placed_structure_of_sequences(type_spec)


if __name__ == "__main__":
  tf.test.main()
