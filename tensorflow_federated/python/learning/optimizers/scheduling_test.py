# Copyright 2022, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.learning.optimizers import adagrad
from tensorflow_federated.python.learning.optimizers import adam
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import rmsprop
from tensorflow_federated.python.learning.optimizers import scheduling
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.optimizers import yogi


@tf.function
def _example_schedule_fn(round_num):
  if round_num < 2:
    return 0.1
  return 0.01


class ScheduledLROptimizerTest(parameterized.TestCase, tf.test.TestCase):

  def test_scheduled_sgd_computes_correctly(self):
    scheduled_sgd = scheduling.schedule_learning_rate(
        sgdm.build_sgdm(1.0), _example_schedule_fn
    )

    weight = tf.constant(1.0)
    gradient = tf.constant(1.0)
    state = scheduled_sgd.initialize(tf.TensorSpec((), tf.float32))
    state, weight = scheduled_sgd.next(state, weight, gradient)
    self.assertAllClose(0.9, weight)  # Learning rate initially 0.1.
    state, weight = scheduled_sgd.next(state, weight, gradient)
    self.assertAllClose(0.8, weight)
    state, weight = scheduled_sgd.next(state, weight, gradient)
    self.assertAllClose(0.79, weight)  # Learning rate has decreased to 0.01.
    state, weight = scheduled_sgd.next(state, weight, gradient)
    self.assertAllClose(0.78, weight)

  @parameterized.named_parameters(
      ('adagrad', adagrad.build_adagrad(1.0)),
      ('adam', adam.build_adam(1.0)),
      ('rmsprop', rmsprop.build_rmsprop(1.0)),
      ('sgd', sgdm.build_sgdm(1.0)),
      ('sgdm', sgdm.build_sgdm(1.0, momentum=0.9)),
      ('yogi', yogi.build_yogi(1.0)),
  )
  def test_schedule_learning_rate_integrates_with(self, optimizer):
    scheduled_optimizer = scheduling.schedule_learning_rate(
        optimizer, _example_schedule_fn
    )
    self.assertIsInstance(scheduled_optimizer, optimizer_base.Optimizer)

  def test_keras_optimizer_raises(self):
    keras_optimizer = tf.keras.optimizers.SGD(1.0)
    with self.assertRaises(TypeError):
      scheduling.schedule_learning_rate(keras_optimizer, _example_schedule_fn)

  def test_scheduling_scheduled_optimizer_raises(self):
    scheduled_optimizer = scheduling.schedule_learning_rate(
        sgdm.build_sgdm(1.0), _example_schedule_fn
    )
    twice_scheduled_optimizer = scheduling.schedule_learning_rate(
        scheduled_optimizer, _example_schedule_fn
    )
    with self.assertRaisesRegex(KeyError, 'must have learning rate'):
      twice_scheduled_optimizer.initialize(tf.TensorSpec((), tf.float32))


if __name__ == '__main__':
  tf.test.main()
