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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Tests for data_backend_example_bindings."""

import asyncio

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from pybind11_abseil import status
from tensorflow_federated.examples.custom_data_backend import data_backend_example
from tensorflow_federated.proto.v0 import computation_pb2 as tff_computation_proto

# These values are defined in data_backend_example.cc.
STRING_URI = 'string'
STRING_VALUE = b'fooey'
INT_STRUCT_URI = 'int_struct'
INT_VALUE = 55


class DataBackendExampleTest(tff.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('string', STRING_URI, tf.string, STRING_VALUE),
      ('int_struct', INT_STRUCT_URI,
       (tf.int32,), tff.structure.Struct.unnamed(INT_VALUE)),
  )
  def test_materialize_returns(self, uri, type_signature, expected_value):
    backend = data_backend_example.DataBackendExample()
    value = asyncio.run(
        backend.materialize(
            tff_computation_proto.Data(uri=uri), tff.to_type(type_signature)))
    self.assertEqual(value, expected_value)

  def test_raises_no_uri(self):
    backend = data_backend_example.DataBackendExample()
    with self.assertRaisesRegex(status.StatusNotOk, 'non-URI data blocks'):
      asyncio.run(
          backend.materialize(tff_computation_proto.Data(), tff.to_type(())))

  def test_raises_unknown_uri(self):
    backend = data_backend_example.DataBackendExample()
    with self.assertRaisesRegex(status.StatusNotOk, 'Unknown URI'):
      asyncio.run(
          backend.materialize(
              tff_computation_proto.Data(uri='unknown_uri'), tff.to_type(())))


if __name__ == '__main__':
  tff.test.main()
