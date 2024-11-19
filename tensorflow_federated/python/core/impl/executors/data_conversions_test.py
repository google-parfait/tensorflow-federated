# Copyright 2021, The TensorFlow Federated Authors.
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

from absl.testing import absltest
import federated_language

from tensorflow_federated.python.core.impl.executors import data_conversions


class DataConversionsTest(absltest.TestCase):

  def test_converts_placement_keyed_to_string_keyed(self):
    num_clients = 10
    placement_keyed_mapping = {
        federated_language.SERVER: 1,
        federated_language.CLIENTS: num_clients,
    }
    expected_string_keyed_mapping = {
        federated_language.SERVER.uri: 1,
        federated_language.CLIENTS.uri: num_clients,
    }

    string_keyed_mapping = (
        data_conversions.convert_cardinalities_dict_to_string_keyed(
            placement_keyed_mapping
        )
    )

    self.assertEqual(string_keyed_mapping, expected_string_keyed_mapping)

  def test_raises_string_keyed_mapping(self):
    string_keyed_mapping = {
        federated_language.SERVER.uri: 1,
        federated_language.CLIENTS.uri: 5,
    }

    with self.assertRaises(TypeError):
      data_conversions.convert_cardinalities_dict_to_string_keyed(
          string_keyed_mapping
      )

  def test_raises_non_integer_values(self):
    placement_keyed_non_integer_valued_mapping = {
        federated_language.SERVER: 1.0,
        federated_language.CLIENTS: 10.0,
    }

    with self.assertRaises(TypeError):
      data_conversions.convert_cardinalities_dict_to_string_keyed(
          placement_keyed_non_integer_valued_mapping
      )


if __name__ == '__main__':
  absltest.main()
