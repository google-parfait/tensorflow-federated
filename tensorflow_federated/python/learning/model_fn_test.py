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
"""Tests for computations.model_fn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf
from tensorflow_federated.python.learning import model_fn


class MinimalModelFn(model_fn.ModelFn):

  def build(self):
    tf_examples = model_fn.model_input_tensor()
    x = tf.parse_example(tf_examples,
                         {'x': tf.FixedLenFeature([], tf.float32),})['x']
    metric_var = tf.get_variable('metric_var', [],
                                 initializer=tf.zeros_initializer())
    x_sum = tf.reduce_sum(x)
    return model_fn.ModelSpec(
        loss=(2.0 * x_sum),
        minibatch_update_ops=[tf.assign_add(metric_var, x_sum)],
        metrics=[model_fn.Metric.sum('SumX', metric_var)])


def _make_tf_example(x):
  return tf.train.Example(
      features=tf.train.Features(feature={
          'x': tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
      })).SerializeToString()


class ModelFnTest(tf.test.TestCase):

  def test_minimal_model_fn(self):
    input_tensor = tf.constant([
        _make_tf_example(x) for x in [1.0, 2.0]])
    with model_fn.set_model_input_tensor(input_tensor):
      model_spec = MinimalModelFn().build()
    self.assertIsInstance(model_spec, model_fn.ModelSpec)
    metric = model_spec.metrics[0]
    self.assertEqual(metric.name, 'SumX')
    self.assertEqual(metric.aggregation_spec.compute_sum, True)
    self.assertEqual(metric.aggregation_spec.compute_average_with_weight, None)

    with self.test_session() as sess:
      sess.run([tf.local_variables_initializer(),
                tf.global_variables_initializer()])
      self.assertEqual(sess.run(model_spec.loss), 6.0)

      sess.run(model_spec.minibatch_update_ops)
      metrics = sess.run({m.name: m.value_tensor for m in model_spec.metrics})
      self.assertItemsEqual(metrics.keys(), ['SumX'])
      self.assertEqual(metrics['SumX'], 3.0)


if __name__ == '__main__':
  tf.test.main()
