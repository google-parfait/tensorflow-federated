# Copyright 2019, The TensorFlow Federated Authors.
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

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.learning import federated_evaluation
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.learning import model_utils


class TestModel(model.Model):

  def __init__(self):
    self._variables = collections.namedtuple('Vars', 'max_temp num_over')(
        max_temp=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=[]),
            name='max_temp',
            trainable=True),
        num_over=tf.Variable(0.0, name='num_over', trainable=False))

  @property
  def trainable_variables(self):
    return [self._variables.max_temp]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [self._variables.num_over]

  @property
  def input_spec(self):
    return collections.OrderedDict([('temp', tf.TensorSpec([None],
                                                           tf.float32))])

  @tf.function
  def forward_pass(self, batch, training=True):
    assert not training
    num_over = tf.reduce_sum(
        tf.cast(
            tf.greater(batch['temp'], self._variables.max_temp), tf.float32))
    self._variables.num_over.assign_add(num_over)
    loss = tf.constant(0.0)
    predictions = tf.zeros_like(batch['temp'])
    return model.BatchOutput(
        loss=loss,
        predictions=predictions,
        num_examples=tf.shape(predictions)[0])

  @tf.function
  def report_local_outputs(self):
    return collections.OrderedDict([('num_over', self._variables.num_over)])

  @property
  def federated_output_computation(self):

    def aggregate_metrics(client_metrics):
      return collections.OrderedDict(
          num_over=intrinsics.federated_sum(client_metrics.num_over))

    return computations.federated_computation(aggregate_metrics)


class FederatedEvaluationTest(test.TestCase):

  def test_federated_evaluation(self):
    evaluate = federated_evaluation.build_federated_evaluation(TestModel)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<<trainable=<float32>,non_trainable=<>>@SERVER,'
        '{<temp=float32[?]>*}@CLIENTS> -> <num_over=float32@SERVER>)')

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [5.0]),
            ('non_trainable', []),
        ]), [
            [_temp_dict([1.0, 10.0, 2.0, 7.0]),
             _temp_dict([6.0, 11.0])],
            [_temp_dict([9.0, 12.0, 13.0])],
            [_temp_dict([1.0]), _temp_dict([22.0, 23.0])],
        ])
    self.assertEqual(result, collections.OrderedDict(num_over=9.0))

  def test_federated_evaluation_with_keras(self):

    def model_fn():
      keras_model = tf.keras.Sequential([
          tf.keras.layers.Input(shape=(1,)),
          tf.keras.layers.Dense(
              1,
              kernel_initializer='ones',
              bias_initializer='zeros',
              activation=None)
      ],
                                        name='my_model')
      return keras_utils.from_keras_model(
          keras_model,
          input_spec=collections.OrderedDict(
              x=tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
              y=tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
          ),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.Accuracy()])

    evaluate_comp = federated_evaluation.build_federated_evaluation(model_fn)
    initial_weights = tf.nest.map_structure(
        lambda x: x.read_value(),
        model_utils.enhance(model_fn()).weights)

    def _input_dict(temps):
      return collections.OrderedDict([
          ('x', np.reshape(np.array(temps, dtype=np.float32), (-1, 1))),
          ('y', np.reshape(np.array(temps, dtype=np.float32), (-1, 1))),
      ])

    result = evaluate_comp(
        initial_weights,
        [[_input_dict([1.0, 10.0, 2.0, 7.0]),
          _input_dict([6.0, 11.0])], [_input_dict([9.0, 12.0, 13.0])],
         [_input_dict([1.0]), _input_dict([22.0, 23.0])]])
    # Expect 100% accuracy and no loss because we've constructed the identity
    # function and have the same x's and y's for training data.
    self.assertDictEqual(result,
                         collections.OrderedDict(accuracy=1.0, loss=0.0))


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
