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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs.model_compression.core import encoding_stage
from tensorflow_federated.python.tensorflow_libs.model_compression.core import test_utils


class TFStyleEncodeDecodeTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `tf_style_encode` and `tf_style_decode` decorators."""

  _DEFAULT_SCOPE = 'test_variable_scope'

  def _test_encode_fn(self, default_scope):
    """Returns decorated test encode method."""
    return encoding_stage.tf_style_encode(
        default_scope)(lambda _, x, p, name: x + p['param'])

  def _test_decode_fn(self, default_scope):
    """Returns decorated test decode method."""
    return encoding_stage.tf_style_decode(default_scope)(
        lambda _, x, p, shape, name: tf.reshape(x['val'] + p['param'], shape))

  @parameterized.parameters(None, 'different_test_variable_scope')
  def test_encode_decorator(self, name):
    """Test encode decorator works as expected."""
    test_encode_fn = self._test_encode_fn(self._DEFAULT_SCOPE)
    encoded_x = self.evaluate(test_encode_fn(None, 2.5, {'param': 10.0}, name))

    # The graph should contain three nodes. The two above Python constants
    # converted to a Tensor object, and the resulting sum.
    self.assertLen(tf.get_default_graph().as_graph_def().node, 3)
    for node in tf.get_default_graph().as_graph_def().node:
      # All nodes should be enclosed in appropriate scope.
      self.assertIn(self._DEFAULT_SCOPE if name is None else name, node.name)
    # The functionality (sum) is not modified.
    self.assertEqual(12.5, encoded_x)

  def test_encode_decorator_different_graphs(self):
    """Input Tensors from different tf.Graph instances should raise an error."""
    # The test method should not actually use the input valueas, to ensure the
    # error is not raised in a different way.
    test_encode_fn = encoding_stage.tf_style_encode(
        self._DEFAULT_SCOPE)(lambda _, x, p, name: tf.constant(0.0))
    graph_1, graph_2 = tf.Graph(), tf.Graph()
    with graph_1.as_default():
      x = tf.constant(2.5)
    with graph_2.as_default():
      params = {'param': tf.constant(10.0)}
    with self.assertRaises(ValueError):
      self.evaluate(test_encode_fn(None, x, params, None))

  @parameterized.parameters(None, 'different_test_variable_scope')
  def test_decode_decorator(self, name):
    """Test decode decorator works as expected."""
    test_decode_fn = self._test_decode_fn(self._DEFAULT_SCOPE)
    decoded_x = self.evaluate(
        test_decode_fn(None,
                       {'val': np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)},
                       {'param': 1.0}, [4], name))

    # The graph should contain five nodes. The three above Python constants
    # converted to a Tensor object, the subtraction, and the final reshape.
    self.assertLen(tf.get_default_graph().as_graph_def().node, 5)
    for node in tf.get_default_graph().as_graph_def().node:
      # All nodes should be enclosed in appropriate scope.
      self.assertIn(self._DEFAULT_SCOPE if name is None else name, node.name)
    # The functionality (sum + reshape) is not modified.
    self.assertAllEqual(np.array([2.0, 3.0, 4.0, 5.0]), decoded_x)

  def test_decode_decorator_different_graphs(self):
    """Input Tensors from different tf.Graph instances should raise an error."""
    # The test method should not actually use the input valueas, to ensure the
    # error is not raised in a different way.
    test_decode_fn = encoding_stage.tf_style_decode(
        self._DEFAULT_SCOPE)(lambda _, x, p, shape, name: tf.constant(0.0))
    graph_1, graph_2 = tf.Graph(), tf.Graph()
    with graph_1.as_default():
      x = {'val': tf.constant(2.5)}
    with graph_2.as_default():
      params = {'param': tf.constant(10.0)}
    with self.assertRaises(ValueError):
      self.evaluate(test_decode_fn(None, x, params, [], None))


class NoneStateAdaptiveEncodingStageTest(tf.test.TestCase,
                                         parameterized.TestCase):

  def test_as_adaptive_encoding_stage(self):
    """Tests correctness of the wrapped encoding stage."""
    a_var = tf.get_variable('a', initializer=2.0)
    b_var = tf.get_variable('b', initializer=3.0)
    stage = test_utils.SimpleLinearEncodingStage(a_var, b_var)
    wrapped_stage = encoding_stage.as_adaptive_encoding_stage(stage)
    self.assertIsInstance(wrapped_stage,
                          encoding_stage.AdaptiveEncodingStageInterface)

    x = tf.constant(2.0)
    state = wrapped_stage.initial_state()
    encode_params, decode_params = wrapped_stage.get_params(state)
    encoded_x, state_update_tensors = wrapped_stage.encode(x, encode_params)
    updated_state = wrapped_stage.update_state(state, state_update_tensors)
    decoded_x = wrapped_stage.decode(encoded_x, decode_params)

    # Test that the added state functionality is empty.
    self.assertDictEqual({}, state)
    self.assertDictEqual({}, state_update_tensors)
    self.assertDictEqual({}, updated_state)
    self.assertDictEqual({}, wrapped_stage.state_update_aggregation_modes)
    # Test that __getattr__ retrieves attributes of the wrapped stage.
    self.assertIsInstance(wrapped_stage._a, tf.Variable)
    self.assertIs(wrapped_stage._a, a_var)
    self.assertIsInstance(wrapped_stage._b, tf.Variable)
    self.assertIs(wrapped_stage._b, b_var)

    # Test the functionality remain unchanged.
    self.assertEqual(stage.compressible_tensors_keys,
                     wrapped_stage.compressible_tensors_keys)
    self.assertEqual(stage.commutes_with_sum, wrapped_stage.commutes_with_sum)
    self.assertEqual(stage.decode_needs_input_shape,
                     wrapped_stage.decode_needs_input_shape)

    self.evaluate(tf.global_variables_initializer())
    test_data = test_utils.TestData(*self.evaluate([x, encoded_x, decoded_x]))
    self.assertEqual(2.0, test_data.x)
    self.assertEqual(7.0, test_data.encoded_x['values'])
    self.assertEqual(2.0, test_data.decoded_x)

  def test_as_adaptive_encoding_stage_identity(self):
    """Tests that this acts as identity for an adaptive encoding stage."""
    adaptive_stage = encoding_stage.NoneStateAdaptiveEncodingStage(
        test_utils.PlusOneEncodingStage())
    wrapped_stage = encoding_stage.as_adaptive_encoding_stage(adaptive_stage)
    self.assertIs(adaptive_stage, wrapped_stage)

  @parameterized.parameters(1.0, 'string', object)
  def test_as_adaptive_encoding_stage_raises(self, not_a_stage):
    with self.assertRaises(TypeError):
      encoding_stage.as_adaptive_encoding_stage(not_a_stage)


if __name__ == '__main__':
  tf.test.main()
