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
"""Tests for model_utils."""
import collections

import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils


class TestModel(model_lib.Model):
  """A very simple test model for testing type signatures."""

  def __init__(self):
    self.w = tf.Variable([0.0, 0.0, 0.0], name='w')
    self.b = tf.Variable([0.0], name='b')
    self.c = tf.Variable(0, name='c')
    self.num_examples = tf.Variable(0, name='num_examples', trainable=False)

  @property
  def trainable_variables(self):
    return [self.w, self.b]

  @property
  def non_trainable_variables(self):
    return [self.c]

  @property
  def local_variables(self):
    return [self.num_examples]

  @property
  def input_spec(self):
    return computation_types.StructType((
        computation_types.TensorSpec(tf.float32, [3]),
        computation_types.TensorSpec(tf.float32, [1]),
    ))

  def predict_on_batch(self, batch_input, training=True):
    del training  # Unused.
    del batch_input  # Unused.
    return 1.0

  def forward_pass(self, batch_input, training=True):
    return model_lib.BatchOutput(
        loss=0.0, predictions=self.predict_on_batch, num_examples=0)

  def report_local_outputs(self):
    return [self.num_examples.read_value()]

  @property
  def federated_output_computation(self):
    return computations.federated_computation(lambda x: x)


class WeightsTypeFromModelTest(test_case.TestCase):

  def test_fails_not_callable_or_model(self):
    with self.assertRaises(TypeError):
      model_utils.weights_type_from_model(0)
    with self.assertRaises(TypeError):
      model_utils.weights_type_from_model(lambda: 0)

  def test_returns_model_weights_for_model(self):
    model = TestModel()
    weights_type = model_utils.weights_type_from_model(model)
    self.assertEqual(
        computation_types.StructWithPythonType(
            [('trainable',
              computation_types.StructWithPythonType([
                  computation_types.TensorType(tf.float32, [3]),
                  computation_types.TensorType(tf.float32, [1]),
              ], list)),
             ('non_trainable',
              computation_types.StructWithPythonType([
                  computation_types.TensorType(tf.int32),
              ], list))], model_utils.ModelWeights), weights_type)

  def test_returns_model_weights_for_model_callable(self):
    weights_type = model_utils.weights_type_from_model(TestModel)
    self.assertEqual(
        computation_types.StructWithPythonType(
            [('trainable',
              computation_types.StructWithPythonType([
                  computation_types.TensorType(tf.float32, [3]),
                  computation_types.TensorType(tf.float32, [1]),
              ], list)),
             ('non_trainable',
              computation_types.StructWithPythonType([
                  computation_types.TensorType(tf.int32),
              ], list))], model_utils.ModelWeights), weights_type)


class ParamCountFromModelTest(test_case.TestCase):

  def test_fails_not_model(self):
    with self.assertRaises(TypeError):
      model_utils.parameter_count_from_model(0)
    with self.assertRaises(TypeError):
      model_utils.parameter_count_from_model(lambda: 0)

  def test_returns_model_weights_for_model(self):
    model = TestModel()
    parameter_count_dict = model_utils.parameter_count_from_model(model)
    expected_parameter_count = collections.OrderedDict(
        num_tensors=2, parameters=4, num_unspecified_tensors=0)
    self.assertEqual(expected_parameter_count, parameter_count_dict)

  def test_returns_model_weights_for_model_callable(self):
    parameter_count_dict = model_utils.parameter_count_from_model(TestModel)
    expected_parameter_count = collections.OrderedDict(
        num_tensors=2, parameters=4, num_unspecified_tensors=0)
    self.assertEqual(expected_parameter_count, parameter_count_dict)


if __name__ == '__main__':
  test_case.main()
