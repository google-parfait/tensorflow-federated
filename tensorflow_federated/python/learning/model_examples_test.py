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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.learning import model_examples


class ModelExamplesTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('', 1), ('_three_features', 3))
  def test_linear_regression(self, feature_dim):
    model = model_examples.LinearRegression(feature_dim=feature_dim)
    batch = model.make_batch(
        x=tf.constant([[0.0] * feature_dim, [1.0] * feature_dim]),
        y=tf.constant([[0.0], [1.0]]))

    output = model.forward_pass(batch)
    metrics = model.report_local_outputs()

    self.assertAllEqual(output.predictions, [[0.0], [0.0]])
    # The residuals are (0., 1.), so average loss is 0.5 * 0.5 * 1.
    self.assertEqual(output.loss, 0.25)
    self.assertDictEqual(
        metrics, {
            'num_examples': 2,
            'num_examples_float': 2.0,
            'num_batches': 1,
            'loss': 0.25
        })

  def test_tff(self):
    feature_dim = 2

    @computations.tf_computation
    def forward_pass_and_output():
      model = model_examples.LinearRegression(feature_dim)

      @tf.function
      def _train(batch):
        batch_output = model.forward_pass(batch)
        local_output = model.report_local_outputs()
        return batch_output, local_output

      return _train(
          batch=model.make_batch(
              x=tf.constant([[0.0, 0.0], [1.0, 1.0]]),
              y=tf.constant([[0.0], [1.0]])))

    batch_output, local_output = forward_pass_and_output()
    self.assertAllEqual(batch_output.predictions, [[0.0], [0.0]])
    self.assertEqual(batch_output.loss, 0.25)
    self.assertDictEqual(
        local_output, {
            'num_examples': 2,
            'num_batches': 1,
            'loss': 0.25,
            'num_examples_float': 2.0,
        })

    # TODO(b/122114585): Add tests for model.federated_output_computation.


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
