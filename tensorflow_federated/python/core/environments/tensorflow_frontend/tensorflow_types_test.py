# Copyright 2018, The TensorFlow Federated Authors.
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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types


class TensorflowToTypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'dtype',
          tf.int32,
          federated_language.TensorType(np.int32),
      ),
      (
          'dtype_nested',
          [tf.int32],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32),
              ],
              list,
          ),
      ),
      (
          'dtype_mixed',
          [tf.int32, np.float32],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32),
                  federated_language.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'tensor_like_shape_fully_defined',
          (tf.int32, tf.TensorShape([2, 3])),
          federated_language.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_like_shape_partially_defined',
          (tf.int32, tf.TensorShape([2, None])),
          federated_language.TensorType(np.int32, shape=[2, None]),
      ),
      (
          'tensor_like_shape_unknown',
          (tf.int32, tf.TensorShape(None)),
          federated_language.TensorType(np.int32, shape=None),
      ),
      (
          'tensor_like_shape_scalar',
          (tf.int32, tf.TensorShape([])),
          federated_language.TensorType(np.int32),
      ),
      (
          'tensor_like_dtype_only',
          (tf.int32, [2, 3]),
          federated_language.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_like_shape_only',
          (np.int32, tf.TensorShape([2, 3])),
          federated_language.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_like_nested',
          [(tf.int32, tf.TensorShape([2, 3]))],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32, shape=[2, 3]),
              ],
              list,
          ),
      ),
      (
          'tensor_like_mixed',
          [(tf.int32, tf.TensorShape([2, 3])), np.float32],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32, shape=[2, 3]),
                  federated_language.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'tensor_spec',
          tf.TensorSpec(shape=[2, 3], dtype=tf.int32),
          federated_language.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_spec_nested',
          [tf.TensorSpec(shape=[2, 3], dtype=tf.int32)],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32, shape=[2, 3]),
              ],
              list,
          ),
      ),
      (
          'tensor_spec_mixed',
          [tf.TensorSpec(shape=[2, 3], dtype=tf.int32), np.float32],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32, shape=[2, 3]),
                  federated_language.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'dataset_spec',
          tf.data.DatasetSpec(tf.TensorSpec(shape=[2, 3], dtype=tf.int32)),
          federated_language.SequenceType(
              federated_language.TensorType(np.int32, shape=[2, 3])
          ),
      ),
      (
          'dataset_spec_nested',
          [
              tf.data.DatasetSpec(tf.TensorSpec(shape=[2, 3], dtype=tf.int32)),
          ],
          federated_language.StructWithPythonType(
              [
                  federated_language.SequenceType(
                      federated_language.TensorType(np.int32, shape=[2, 3])
                  ),
              ],
              list,
          ),
      ),
      (
          'dataset_spec_mixed',
          [
              tf.data.DatasetSpec(tf.TensorSpec(shape=[2, 3], dtype=tf.int32)),
              np.float32,
          ],
          federated_language.StructWithPythonType(
              [
                  federated_language.SequenceType(
                      federated_language.TensorType(np.int32, shape=[2, 3])
                  ),
                  federated_language.TensorType(np.float32),
              ],
              list,
          ),
      ),
  )
  def test_returns_result_with_tensorflow_obj(self, obj, expected_result):
    actual_result = tensorflow_types.to_type(obj)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('type', federated_language.TensorType(np.int32)),
      ('dtype', np.int32),
      ('tensor_like', (np.int32, [2, 3])),
      ('sequence_unnamed', [np.float64, np.int32, np.str_]),
      ('sequence_named', [('a', np.float64), ('b', np.int32), ('c', np.str_)]),
      ('mapping', {'a': np.float64, 'b': np.int32, 'c': np.str_}),
  )
  def test_delegates_result_with_obj(self, obj):

    with mock.patch.object(
        federated_language, 'to_type', autospec=True, spec_set=True
    ) as mock_to_type:
      tensorflow_types.to_type(obj)
      mock_to_type.assert_called_once_with(obj)


if __name__ == '__main__':
  absltest.main()
