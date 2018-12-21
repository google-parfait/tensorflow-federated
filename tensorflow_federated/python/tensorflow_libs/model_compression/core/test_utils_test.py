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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs.model_compression.core import test_utils


class PlusOneEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.PlusOneEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertAllClose(data.x + 1.0, data.encoded_x['values'])


class TimesTwoEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.TimesTwoEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertAllClose(data.x * 2.0, data.encoded_x['values'])


class SimpleLinearEncodingStageTest(test_utils.BaseEncodingStageTest):

  _DEFAULT_A = 2.0
  _DEFAULT_B = 3.0

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.SimpleLinearEncodingStage(self._DEFAULT_A,
                                                self._DEFAULT_B)

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertAllClose(data.x * self._DEFAULT_A + self._DEFAULT_B,
                        data.encoded_x['values'])

  def test_basic_encode_decode_tf_constructor_parameters(self):
    """Tests the core funcionality with `tf.Variable` constructor parameters."""
    a_var = tf.get_variable('a', initializer=self._DEFAULT_A)
    b_var = tf.get_variable('b', initializer=self._DEFAULT_B)
    stage = test_utils.SimpleLinearEncodingStage(a_var, b_var)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
    x = self.default_input()
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = self.evaluate(test_utils.TestData(x, encoded_x, decoded_x))
    self.common_asserts_for_test_data(test_data)

    # Change the variables and verify the behavior of stage changes.
    self.evaluate([tf.assign(a_var, 5.0), tf.assign(b_var, 6.0)])
    test_data = self.evaluate(test_utils.TestData(x, encoded_x, decoded_x))
    self.assertAllClose(test_data.x * 5.0 + 6.0, test_data.encoded_x['values'])


class ReduceMeanEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.ReduceMeanEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(np.tile(np.mean(data.x), data.x.shape), data.decoded_x)
    self.assertAllClose(
        np.mean(data.x, keepdims=True), data.encoded_x['values'])

  def test_one_to_many_with_unknown_shape(self):
    """Tests that encoding works with statically not known input shape."""
    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(), test_utils.get_tensor_with_random_shape)
    self.common_asserts_for_test_data(test_data)

  @parameterized.parameters([2], [2, 3], [2, 3, 4], [2, 3, 4, 5])
  def test_one_to_many_with_multiple_input_shapes(self, *shape):
    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(), lambda: tf.random_uniform(shape))
    self.common_asserts_for_test_data(test_data)


class RandomAddSubtractOneEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.RandomAddSubtractOneEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.normal([5])

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    for x, decoded_x in zip(data.x, data.decoded_x):
      self.assertTrue(
          np.isclose(decoded_x, x - 1) or np.isclose(decoded_x, x) or
          np.isclose(decoded_x, x + 1))
    self.assertAllEqual(data.encoded_x['values'], data.decoded_x)

  def test_approximately_unbiased_in_expectation(self):
    """Tests that average of encodings is more accurate than a single one."""
    # Use a constant input value.
    x = self.evaluate(self.default_input())
    stage = self.default_encoding_stage()
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = []
    for _ in range(100):
      test_data.append(
          test_utils.TestData(
              *self.evaluate_tf_py_list([x, encoded_x, decoded_x])))

    # Check that the average error created by encoding is significantly larger
    # than error of average of encodings. This is an simple (imperfect)
    # empirical check that the encoding is unbiased.
    mean_error = np.mean([np.linalg.norm(x - d.decoded_x) for d in test_data])
    error_of_mean = np.linalg.norm(
        x - np.mean([d.decoded_x for d in test_data], axis=0))
    self.assertGreater(mean_error, error_of_mean * 5)


class TestUtilsTest(tf.test.TestCase):
  """Tests for other utilities in `test_utils.py`."""

  def test_get_tensor_with_random_shape(self):
    x = test_utils.get_tensor_with_random_shape()
    self.assertIsInstance(x, tf.Tensor)
    self.assertFalse(x.shape.is_fully_defined())

    # Assert that unknown shape corresponds to a value of actually random shape
    # at execution time.
    samples = [self.evaluate(x) for _ in range(10)]
    self.assertGreater(len(set([len(s) for s in samples])), 1)

    # Test that source_fn has effect on the ourpur values.
    x_uniform = test_utils.get_tensor_with_random_shape(
        expected_num_elements=50, source_fn=tf.random.uniform)
    x_normal = test_utils.get_tensor_with_random_shape(
        expected_num_elements=50, source_fn=tf.random.normal)
    self.assertGreaterEqual(self.evaluate(tf.reduce_min(x_uniform)), 0.0)
    self.assertLess(self.evaluate(tf.reduce_min(x_normal)), 0.0)


if __name__ == '__main__':
  tf.test.main()
