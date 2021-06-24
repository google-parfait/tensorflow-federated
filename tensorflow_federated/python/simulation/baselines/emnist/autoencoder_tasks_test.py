# Copyright 2021, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines.emnist import autoencoder_tasks


class CreateAutoencoderTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('emnist_10', True),
      ('emnist_62', False),
  )
  def test_constructs_with_eval_client_spec(self, only_digits):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=2, max_elements=5, shuffle_buffer_size=10)
    baseline_task_spec = autoencoder_tasks.create_autoencoder_task(
        train_client_spec,
        eval_client_spec=eval_client_spec,
        only_digits=only_digits,
        use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

  @parameterized.named_parameters(
      ('emnist_10', True),
      ('emnist_62', False),
  )
  def test_constructs_with_no_eval_client_spec(self, only_digits):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    baseline_task_spec = autoencoder_tasks.create_autoencoder_task(
        train_client_spec, only_digits=only_digits, use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
