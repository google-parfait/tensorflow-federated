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
import portpicker

from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import executor_test_utils_bindings


class ReferenceResolvingExecutorBindingsTest(absltest.TestCase):

  def test_construction(self):
    mock_executor = executor_test_utils_bindings.create_mock_executor()
    try:
      executor_bindings.create_reference_resolving_executor(mock_executor)
    except Exception:  # pylint: disable=broad-except
      self.fail('Raised `Exception` unexpectedly.')


class ComposingExecutorBindingsTest(absltest.TestCase):

  def test_construction(self):
    mock_server_executor = executor_test_utils_bindings.create_mock_executor()
    mock_child_executor = executor_test_utils_bindings.create_mock_executor()
    children = [
        executor_bindings.create_composing_child(
            mock_child_executor,
            {federated_language.CLIENTS: 0},
        )
    ]
    try:
      executor_bindings.create_composing_executor(
          mock_server_executor, children
      )
    except Exception:  # pylint: disable=broad-except
      self.fail('Raised `Exception` unexpectedly.')


class SequenceExecutorBindingsTest(absltest.TestCase):

  def test_construction(self):
    mock_executor = executor_test_utils_bindings.create_mock_executor()
    try:
      executor_bindings.create_sequence_executor(mock_executor)
    except Exception:  # pylint: disable=broad-except
      self.fail('Raised `Exception` unexpectedly.')


class FederatingExecutorBindingsTest(absltest.TestCase):

  def test_construction(self):
    mock_server_executor = executor_test_utils_bindings.create_mock_executor()
    mock_client_executor = executor_test_utils_bindings.create_mock_executor()
    cardinalities = {federated_language.CLIENTS: 0}

    try:
      executor_bindings.create_federating_executor(
          mock_server_executor, mock_client_executor, cardinalities
      )
    except Exception:  # pylint: disable=broad-except
      self.fail('Raised `Exception` unexpectedly.')


class RemoteExecutorBindingsTest(absltest.TestCase):

  def test_construction_with_insecure_channel(self):
    channel = executor_bindings.create_insecure_grpc_channel(
        'localhost:{}'.format(portpicker.pick_unused_port())
    )
    try:
      executor_bindings.create_remote_executor(
          channel,
          cardinalities={federated_language.CLIENTS: 10},
      )
    except Exception:  # pylint: disable=broad-except
      self.fail('Raised `Exception` unexpectedly.')


if __name__ == '__main__':
  absltest.main()
