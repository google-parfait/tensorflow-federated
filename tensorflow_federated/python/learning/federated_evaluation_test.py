# Lint as: python3
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
"""Tests for federated_evaluation.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import federated_evaluation
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
        tf.to_float(tf.greater(batch['temp'], self._variables.max_temp)))
    tf.assign_add(self._variables.num_over, num_over)
    loss = tf.constant(0.0)
    predictions = tf.zeros_like(batch['temp'])
    return model.BatchOutput(loss=loss, predictions=predictions)

  @tf.function
  def report_local_outputs(self):
    return collections.OrderedDict([('num_over', self._variables.num_over)])

  @property
  def federated_output_computation(self):
    return tff.federated_computation(
        lambda metrics: {'num_over': tff.federated_sum(metrics.num_over)})


class FederatedEvaluationTest(test.TestCase):

  def test_federated_evaluation(self):
    evaluate = federated_evaluation.build_federated_evaluation(TestModel)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<<trainable=<max_temp=float32>,non_trainable=<>>@SERVER,'
        '{<temp=float32[?]>*}@CLIENTS> -> <num_over=float32@SERVER>)')

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    result = evaluate({
        'trainable': {
            'max_temp': 5.0
        },
        'non_trainable': {}
    }, [[_temp_dict([1.0, 10.0, 2.0, 7.0]),
         _temp_dict([6.0, 11.0])], [_temp_dict([9.0, 12.0, 13.0])],
        [_temp_dict([1.0]), _temp_dict([22.0, 23.0])]])
    self.assertEqual(str(result), '<num_over=9.0>')

  def test_federated_evaluation_with_keras(self):

    def model_fn():
      keras_model = tf.keras.Sequential([
          tf.keras.layers.Dense(
              1,
              kernel_initializer='ones',
              bias_initializer='zeros',
              activation=None)
      ],
                                        name='my_model')
      keras_model.compile(
          loss='mean_squared_error',
          optimizer='sgd',
          metrics=[tf.keras.metrics.Accuracy()])
      return model_utils.from_compiled_keras_model(
          keras_model,
          dummy_batch={
              'x': np.zeros((1, 1), np.float32),
              'y': np.zeros((1, 1), np.float32)
          })

    evaluate_comp = federated_evaluation.build_federated_evaluation(model_fn)
    initial_weights = tf.contrib.framework.nest.map_structure(
        lambda x: x.read_value(),
        model_utils.enhance(model_fn()).weights)

    def _input_dict(temps):
      return {
          'x': np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
          'y': np.reshape(np.array(temps, dtype=np.float32), (-1, 1))
      }

    result = evaluate_comp(
        initial_weights,
        [[_input_dict([1.0, 10.0, 2.0, 7.0]),
          _input_dict([6.0, 11.0])], [_input_dict([9.0, 12.0, 13.0])],
         [_input_dict([1.0]), _input_dict([22.0, 23.0])]])
    # Expect 100% accuracy and no loss because we've constructed the identity
    # function and have the same x's and y's for training data.
    self.assertEqual(str(result), '<accuracy=1.0,loss=0.0>')


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  test.main()
