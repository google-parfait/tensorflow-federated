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
import numpy as np
import portpicker
import tensorflow as tf
import tree

from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import executor_test_utils_bindings
from tensorflow_federated.python.core.impl.types import placements


def _to_python_value(value):
  def _fn(obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return None

  return tree.traverse(_fn, value)


class SerializeTensorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_int32', 1, np.int32),
      ('scalar_float64', 2.0, np.float64),
      ('scalar_string', b'abc', np.str_),
      ('tensor_int32', [1, 2, 3], np.int32),
      ('tensor_float64', [2.0, 4.0, 6.0], np.float64),
      ('tensor_string', [[b'abc', b'xyz']], np.str_),
  )
  def test_serialize(self, input_value, dtype):
    serialized_value = executor_bindings.serialize_tensor_value(
        tf.convert_to_tensor(input_value, dtype)
    )
    tensor_proto = tf.make_tensor_proto(values=0)
    self.assertTrue(serialized_value.tensor.Unpack(tensor_proto))
    actual_value = tf.make_ndarray(tensor_proto)
    actual_value = _to_python_value(actual_value)
    self.assertEqual(actual_value, input_value)

  @parameterized.named_parameters(
      ('scalar_int32', 1, np.int32),
      ('scalar_float64', 2.0, np.float64),
      ('scalar_string', b'abc', np.str_),
      ('tensor_int32', [1, 2, 3], np.int32),
      ('tensor_float64', [2.0, 4.0, 6.0], np.float64),
      ('tensor_string', [[b'abc', b'xyz']], np.str_),
  )
  def test_roundtrip(self, input_value, dtype):
    serialized_value = executor_bindings.serialize_tensor_value(
        tf.convert_to_tensor(input_value, dtype)
    )
    deserialized_value = executor_bindings.deserialize_tensor_value(
        serialized_value
    )
    deserialized_value = _to_python_value(deserialized_value)
    self.assertEqual(deserialized_value, input_value)


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
            {placements.CLIENTS: 0},
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
    cardinalities = {placements.CLIENTS: 0}

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
          cardinalities={placements.CLIENTS: 10},
      )
    except Exception:  # pylint: disable=broad-except
      self.fail('Raised `Exception` unexpectedly.')


if __name__ == '__main__':
  absltest.main()
