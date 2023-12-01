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
"""Tests for end-to-end support of composite tensors.

Composite tensors are TensorFlow structures that consist of multiple different
underlying tensors, as well as potentially some additional metadata.

Currently supported composite tensors include:
- tf.RaggedTensor
- tf.SparseTensor

Future support is planned for:
- tf.IndexedSlices
- tf.Struct
"""

from absl.testing import absltest
import tensorflow as tf
import tensorflow_federated as tff


class CompositeTensorTest(absltest.TestCase):

  def test_ragged_tensor_passes_through_computation(self):
    ragged_tensor = tf.RaggedTensor.from_row_splits([0, 0, 0, 0], [0, 1, 4])
    ragged_tensor_spec = tf.RaggedTensorSpec.from_value(ragged_tensor)

    @tff.tf_computation(ragged_tensor_spec)
    def _foo(obj):
      self.assertIsInstance(obj, tf.RaggedTensor)
      return obj

    result = _foo(ragged_tensor)
    self.assertIsInstance(result, tf.RaggedTensor)

  def test_sparse_tensor_passes_through_computation(self):
    sparse_tensor = tf.SparseTensor(indices=[[1]], values=[2], dense_shape=[5])
    sparse_tensor_spec = tf.SparseTensorSpec.from_value(sparse_tensor)

    @tff.tf_computation(sparse_tensor_spec)
    def _foo(obj):
      self.assertIsInstance(obj, tf.SparseTensor)
      return obj

    result = _foo(sparse_tensor)
    self.assertIsInstance(result, tf.SparseTensor)


if __name__ == '__main__':
  tff.backends.test.set_sync_test_cpp_execution_context()
  absltest.main()
