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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.learning.models import model_examples


class ModelExamplesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('', 1), ('_three_features', 3))
  def test_linear_regression(self, feature_dim):
    model = model_examples.LinearRegression(feature_dim=feature_dim)

    expected_initial_local_variables = [0, 0, 0.0]
    self.assertSequenceEqual(
        model.local_variables, expected_initial_local_variables
    )

    batch = collections.OrderedDict(
        x=tf.constant([[0.0] * feature_dim, [1.0] * feature_dim]),
        y=tf.constant([[0.0], [1.0]]),
    )

    output = model.forward_pass(batch)

    self.assertAllEqual(output.predictions, [[0.0], [0.0]])
    # The residuals are (0., 1.), so average loss is 0.5 * 0.5 * 1.
    self.assertEqual(output.loss, 0.25)
    unfinalized_metrics = model.report_local_unfinalized_metrics()
    self.assertEqual(
        unfinalized_metrics,
        collections.OrderedDict(loss=[0.5, 2.0], num_examples=2),
    )
    finalized_metrics = collections.OrderedDict(
        (metric_name, finalizer(unfinalized_metrics[metric_name]))
        for metric_name, finalizer in model.metric_finalizers().items()
    )
    self.assertEqual(
        finalized_metrics, collections.OrderedDict(loss=0.25, num_examples=2)
    )

    # Ensure reset_metrics works.
    model.reset_metrics()
    self.assertSequenceEqual(
        model.local_variables, expected_initial_local_variables
    )
    unfinalized_metrics = model.report_local_unfinalized_metrics()
    self.assertEqual(
        unfinalized_metrics,
        collections.OrderedDict(loss=[0, 0], num_examples=0),
    )

  def test_tff(self):
    feature_dim = 2

    @tensorflow_computation.tf_computation
    def forward_pass_and_output():
      model = model_examples.LinearRegression(feature_dim)

      @tf.function
      def _train(batch):
        batch_output = model.forward_pass(batch)
        unfinalized_metrics = model.report_local_unfinalized_metrics()
        return batch_output, unfinalized_metrics

      return _train(
          batch=collections.OrderedDict(
              x=tf.constant([[0.0, 0.0], [1.0, 1.0]]),
              y=tf.constant([[0.0], [1.0]]),
          )
      )

    batch_output, unfinalized_metrics = forward_pass_and_output()
    self.assertAllEqual(batch_output.predictions, [[0.0], [0.0]])
    self.assertEqual(batch_output.loss, 0.25)
    self.assertEqual(
        unfinalized_metrics,
        collections.OrderedDict(loss=[0.5, 2.0], num_examples=2),
    )
    model = model_examples.LinearRegression(feature_dim)
    finalized_metrics = collections.OrderedDict(
        (metric_name, finalizer(unfinalized_metrics[metric_name]))
        for metric_name, finalizer in model.metric_finalizers().items()
    )
    self.assertEqual(
        finalized_metrics, collections.OrderedDict(loss=0.25, num_examples=2)
    )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
