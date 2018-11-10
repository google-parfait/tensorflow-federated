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
"""Tests for model_fn_examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_federated.python.learning import model_fn
from tensorflow_federated.python.learning.model_fn_examples import NoFeaturesRegressionModelFn


class ModelFnExamplesTest(tf.test.TestCase):

  def test_no_feature_regression_model_fn(self):
    input_tensor = tf.constant(
        # Labels are 2x + 1, so 5.0 and 1.0.
        [NoFeaturesRegressionModelFn.make_tf_example(x)
         for x in [2.0, 0.0]])
    with model_fn.set_model_input_tensor(input_tensor):
      model_spec = NoFeaturesRegressionModelFn().build()
    self.assertIsInstance(model_spec, model_fn.ModelSpec)
    with tf.Session() as sess:
      sess.run([tf.local_variables_initializer(),
                tf.global_variables_initializer()])
      sess.run(model_spec.minibatch_update_ops)
      metrics = sess.run({m.name: m.value_tensor for m in model_spec.metrics})
    # We make a prediction of 1.0 (due to the constant initializer on
    # the model's variable) on labels 5.0 and 1.0.
    self.assertEqual(metrics,  # Metrics are the average over the batch
                     {'abs_error': 4.0 / 2,   # Average over batch
                      'squared_error': 8.0 / 2,  # 0.5 * 16 / 2
                      'avg_prediction': 1.0,
                      'avg_label': 3.0,
                      'num_examples': 2,
                      'num_minibatches': 1,
                      'label_sum': 6.0})

if __name__ == '__main__':
  tf.test.main()
