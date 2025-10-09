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
from absl.testing import parameterized
import federated_language
from federated_language.proto import computation_pb2
from federated_language_executor import executor_pb2

from tensorflow_federated.python.core.impl.executors import cardinality


class SerializeCardinalitiesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'clients_and_server',
          {
              federated_language.CLIENTS: 10,
              federated_language.SERVER: 1,
          },
          [
              executor_pb2.Cardinality(
                  placement=computation_pb2.Placement(
                      uri=federated_language.CLIENTS.uri
                  ),
                  cardinality=10,
              ),
              executor_pb2.Cardinality(
                  placement=computation_pb2.Placement(
                      uri=federated_language.SERVER.uri
                  ),
                  cardinality=1,
              ),
          ],
      ),
      (
          'clients_only',
          {
              federated_language.CLIENTS: 10,
          },
          [
              executor_pb2.Cardinality(
                  placement=computation_pb2.Placement(
                      uri=federated_language.CLIENTS.uri
                  ),
                  cardinality=10,
              ),
          ],
      ),
  )
  def test_serialize_cardinalities_returns_value(
      self, cardinalities, expected_value
  ):
    actual_value = cardinality.serialize_cardinalities(cardinalities)

    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'clients_and_server',
          [
              executor_pb2.Cardinality(
                  placement=computation_pb2.Placement(
                      uri=federated_language.CLIENTS.uri
                  ),
                  cardinality=10,
              ),
              executor_pb2.Cardinality(
                  placement=computation_pb2.Placement(
                      uri=federated_language.SERVER.uri
                  ),
                  cardinality=1,
              ),
          ],
          {
              federated_language.CLIENTS: 10,
              federated_language.SERVER: 1,
          },
      ),
      (
          'clients_only',
          [
              executor_pb2.Cardinality(
                  placement=computation_pb2.Placement(
                      uri=federated_language.CLIENTS.uri
                  ),
                  cardinality=10,
              ),
          ],
          {
              federated_language.CLIENTS: 10,
          },
      ),
  )
  def test_deserialize_cardinalities_returns_value(
      self, serialized_cardinalities, expected_value
  ):
    actual_value = cardinality.deserialize_cardinalities(
        serialized_cardinalities
    )

    self.assertEqual(actual_value, expected_value)


if __name__ == '__main__':
  absltest.main()
