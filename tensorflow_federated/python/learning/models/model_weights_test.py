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

import collections

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import variable


class TestModel(variable.VariableModel):
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
    return variable.BatchOutput(
        loss=0.0, predictions=self.predict_on_batch, num_examples=0
    )

  @tf.function
  def report_local_unfinalized_metrics(self):
    return collections.OrderedDict(num_examples=self.num_examples.read_value())

  def metric_finalizers(self):
    return collections.OrderedDict(num_examples=tf.function(func=lambda x: x))

  @tf.function
  def reset_metrics(self):
    """Resets metrics variables to initial value."""
    for var in self.local_variables:
      var.assign(tf.zeros_like(var))


class WeightsTypeFromModelTest(absltest.TestCase):

  def test_fails_not_callable_or_model(self):
    with self.assertRaises(TypeError):
      model_weights.weights_type_from_model(0)
    with self.assertRaises(TypeError):
      model_weights.weights_type_from_model(lambda: 0)

  def test_returns_model_weights_for_model(self):
    model = TestModel()
    weights_type = model_weights.weights_type_from_model(model)
    self.assertEqual(
        computation_types.StructWithPythonType(
            [
                (
                    'trainable',
                    computation_types.StructWithPythonType(
                        [
                            computation_types.TensorType(np.float32, [3]),
                            computation_types.TensorType(np.float32, [1]),
                        ],
                        list,
                    ),
                ),
                (
                    'non_trainable',
                    computation_types.StructWithPythonType(
                        [
                            computation_types.TensorType(np.int32),
                        ],
                        list,
                    ),
                ),
            ],
            model_weights.ModelWeights,
        ),
        weights_type,
    )

  def test_returns_model_weights_for_model_callable(self):
    weights_type = model_weights.weights_type_from_model(TestModel)
    self.assertEqual(
        computation_types.StructWithPythonType(
            [
                (
                    'trainable',
                    computation_types.StructWithPythonType(
                        [
                            computation_types.TensorType(np.float32, [3]),
                            computation_types.TensorType(np.float32, [1]),
                        ],
                        list,
                    ),
                ),
                (
                    'non_trainable',
                    computation_types.StructWithPythonType(
                        [
                            computation_types.TensorType(np.int32),
                        ],
                        list,
                    ),
                ),
            ],
            model_weights.ModelWeights,
        ),
        weights_type,
    )


class AssignWeightsToTest(tf.test.TestCase):

  def test_weights_to_keras_model(self):
    keras_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[5]),
        tf.keras.layers.Dense(
            units=1, use_bias=False, kernel_initializer='zeros'
        ),
    ])
    self.assertAllEqual(keras_model.trainable_weights, [tf.zeros([5, 1])])
    model_weights.ModelWeights(
        trainable=(tf.ones([5, 1]),), non_trainable=()
    ).assign_weights_to(keras_model)
    self.assertAllEqual(keras_model.trainable_weights, [tf.ones([5, 1])])

  def test_weights_to_variable_model(self):
    model = TestModel()
    self.assertAllClose(
        model.trainable_variables, [tf.zeros([3]), tf.zeros([1])]
    )
    # Note: the arguments must be `list` type sequences, matching the
    # VariableModel types, otherwise an exception will be raised.
    model_weights.ModelWeights(
        trainable=[tf.ones([3]), tf.ones([1])],
        non_trainable=[tf.ones([], dtype=tf.int32)],
    ).assign_weights_to(model)
    self.assertAllClose(model.trainable_variables, [tf.ones([3]), tf.ones([1])])


class ConvertVariablesToArraysTest(tf.test.TestCase):

  def test_raises_exception_in_graph_context(self):
    w = model_weights.ModelWeights(0.0, 0.0)
    with tf.Graph().as_default():
      with self.assertRaisesRegex(ValueError, 'eager'):
        w.convert_variables_to_arrays()

  def test_raises_exception_in_tf_function(self):
    @tf.function
    def a_tf_function(w):
      return w.convert_variables_to_arrays()

    w = model_weights.ModelWeights(0.0, 0.0)

    with self.assertRaisesRegex(ValueError, r'tf\.function'):
      a_tf_function(w)

  def test_raises_exception_in_tf_function_and_graph_context(self):
    @tf.function
    def a_tf_function(w):
      return w.convert_variables_to_arrays()

    w = model_weights.ModelWeights(0.0, 0.0)

    with tf.Graph().as_default():
      with self.assertRaisesRegex(ValueError, 'eager'):
        a_tf_function(w)

  def test_converts_int(self):
    w = model_weights.ModelWeights(1, 2)
    converted = w.convert_variables_to_arrays()
    self.assertIsInstance(converted.trainable, np.ndarray)
    self.assertIsInstance(converted.non_trainable, np.ndarray)
    self.assertEqual(converted.trainable, 1)
    self.assertEqual(converted.non_trainable, 2)

  def test_converts_float(self):
    w = model_weights.ModelWeights(1.0, 2.0)
    converted = w.convert_variables_to_arrays()
    self.assertIsInstance(converted.trainable, np.ndarray)
    self.assertIsInstance(converted.non_trainable, np.ndarray)
    self.assertEqual(converted.trainable, 1.0)
    self.assertEqual(converted.non_trainable, 2.0)

  def test_converts_tensor(self):
    w = model_weights.ModelWeights(tf.constant(1.0), tf.constant(2.0))
    converted = w.convert_variables_to_arrays()
    self.assertIsInstance(converted.trainable, np.ndarray)
    self.assertIsInstance(converted.non_trainable, np.ndarray)
    self.assertEqual(converted.trainable, 1.0)
    self.assertEqual(converted.non_trainable, 2.0)

  def test_converts_variable(self):
    w = model_weights.ModelWeights(tf.Variable(1.0), tf.Variable(2.0))
    converted = w.convert_variables_to_arrays()
    self.assertIsInstance(converted.trainable, np.ndarray)
    self.assertIsInstance(converted.non_trainable, np.ndarray)
    self.assertEqual(converted.trainable, 1.0)
    self.assertEqual(converted.non_trainable, 2.0)

  def test_converts_ndarray(self):
    w = model_weights.ModelWeights(np.array([1.0]), np.array([2.0, 3.0]))
    converted = w.convert_variables_to_arrays()
    self.assertIsInstance(converted.trainable, np.ndarray)
    self.assertIsInstance(converted.non_trainable, np.ndarray)
    self.assertEqual(converted.trainable, [1.0])
    self.assertAllEqual(converted.non_trainable, [2.0, 3.0])

  def test_converts_heterogeneous_types(self):
    w = model_weights.ModelWeights(
        [1, 2.0, tf.constant(3), tf.Variable(4)], [np.zeros([2, 3])]
    )
    converted = w.convert_variables_to_arrays()
    tf.nest.map_structure(
        lambda item: self.assertIsInstance(item, np.ndarray),
        converted.trainable,
    )
    tf.nest.map_structure(
        lambda item: self.assertIsInstance(item, np.ndarray),
        converted.non_trainable,
    )

  def test_converts_struct(self):
    w = model_weights.ModelWeights(
        structure.Struct.unnamed(1.0), structure.Struct.unnamed(2.0, 3.0)
    )
    converted = w.convert_variables_to_arrays()
    structure.map_structure(
        lambda item: self.assertIsInstance(item, np.ndarray),
        converted.trainable,
    )
    structure.map_structure(
        lambda item: self.assertIsInstance(item, np.ndarray),
        converted.non_trainable,
    )
    self.assertAllEqual(
        structure.to_elements(converted.trainable), [(None, np.array([1.0]))]
    )
    self.assertAllEqual(
        structure.to_elements(converted.non_trainable),
        [(None, np.array([2.0])), (None, np.array([3.0]))],
    )

  def test_converts_heterogeneous_struct(self):
    w = model_weights.ModelWeights(
        structure.Struct.named(
            a=1,
            b=2.0,
            c=tf.constant(3),
            d=tf.Variable(4),
            e=structure.Struct.named(nested_a=5, nested_b=6.0),
        ),
        structure.Struct.unnamed(0.0),
    )
    converted = w.convert_variables_to_arrays()
    structure.map_structure(
        lambda item: self.assertIsInstance(item, np.ndarray),
        converted.trainable,
    )
    self.assertAllEqual(
        structure.to_elements(converted.trainable),
        [
            ('a', 1),
            ('b', 2.0),
            ('c', 3),
            ('d', 4),
            ('e', structure.Struct.named(nested_a=5, nested_b=6.0)),
        ],
    )


if __name__ == '__main__':
  absltest.main()
