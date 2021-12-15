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

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.learning import model_examples


class ModelExamplesTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('', 1), ('_three_features', 3))
  def test_linear_regression(self, feature_dim):
    model = model_examples.LinearRegression(feature_dim=feature_dim)
    batch = collections.OrderedDict(
        x=tf.constant([[0.0] * feature_dim, [1.0] * feature_dim]),
        y=tf.constant([[0.0], [1.0]]))

    output = model.forward_pass(batch)
    metrics = model.report_local_outputs()

    self.assertAllEqual(output.predictions, [[0.0], [0.0]])
    # The residuals are (0., 1.), so average loss is 0.5 * 0.5 * 1.
    self.assertEqual(output.loss, 0.25)
    self.assertEqual(
        metrics,
        collections.OrderedDict(
            num_examples=2, num_examples_float=2.0, num_batches=1, loss=0.25))
    unfinalized_metrics = model.report_local_unfinalized_metrics()
    self.assertEqual(unfinalized_metrics,
                     collections.OrderedDict(loss=[0.5, 2.0], num_examples=2))
    finalized_metrics = collections.OrderedDict(
        (metric_name, finalizer(unfinalized_metrics[metric_name]))
        for metric_name, finalizer in model.metric_finalizers().items())
    self.assertEqual(finalized_metrics,
                     collections.OrderedDict(loss=0.25, num_examples=2))

  def test_tff(self):
    feature_dim = 2

    @computations.tf_computation
    def forward_pass_and_output():
      model = model_examples.LinearRegression(feature_dim)

      @tf.function
      def _train(batch):
        batch_output = model.forward_pass(batch)
        local_output = model.report_local_outputs()
        unfinalized_metrics = model.report_local_unfinalized_metrics()
        return batch_output, local_output, unfinalized_metrics

      return _train(
          batch=collections.OrderedDict(
              x=tf.constant([[0.0, 0.0], [1.0, 1.0]]),
              y=tf.constant([[0.0], [1.0]])))

    batch_output, local_output, unfinalized_metrics = forward_pass_and_output()
    self.assertAllEqual(batch_output.predictions, [[0.0], [0.0]])
    self.assertEqual(batch_output.loss, 0.25)
    self.assertEqual(
        local_output,
        collections.OrderedDict(
            num_examples=2, num_examples_float=2.0, num_batches=1, loss=0.25))
    self.assertEqual(unfinalized_metrics,
                     collections.OrderedDict(loss=[0.5, 2.0], num_examples=2))
    model = model_examples.LinearRegression(feature_dim)
    finalized_metrics = collections.OrderedDict(
        (metric_name, finalizer(unfinalized_metrics[metric_name]))
        for metric_name, finalizer in model.metric_finalizers().items())
    self.assertEqual(finalized_metrics,
                     collections.OrderedDict(loss=0.25, num_examples=2))

    # TODO(b/122114585): Add tests for model.federated_output_computation.

  def test_raise_not_implemented_error(self):
    model = model_examples.LinearRegression(use_metrics_aggregator=True)
    with self.assertRaisesRegex(NotImplementedError, 'Do not implement'):
      model.report_local_outputs()
    with self.assertRaisesRegex(NotImplementedError, 'Do not implement'):
      model.federated_output_computation  # pylint: disable=pointless-statement


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
