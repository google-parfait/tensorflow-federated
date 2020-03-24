# Lint as: python3
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

import contextlib

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shared import optimizer_utils

FLAGS = flags.FLAGS
TEST_CLIENT_FLAG_PREFIX = 'test_client'
TEST_SERVER_FLAG_PREFIX = 'test_server'


@contextlib.contextmanager
def flag_sandbox(flag_value_dict):

  def _set_flags(flag_dict):
    for name, value in flag_dict.items():
      FLAGS[name].value = value

  # Store the current values and override with the new.
  preserved_value_dict = {
      name: FLAGS[name].value for name in flag_value_dict.keys()
  }
  _set_flags(flag_value_dict)
  yield

  # Restore the saved values.
  for name in preserved_value_dict.keys():
    FLAGS[name].unparse()
  _set_flags(preserved_value_dict)


def setUpModule():
  # Create flags here to ensure duplicate flags are not created.
  optimizer_utils.define_optimizer_flags(TEST_SERVER_FLAG_PREFIX)
  optimizer_utils.define_optimizer_flags(TEST_CLIENT_FLAG_PREFIX)
  optimizer_utils.define_lr_schedule_flags(TEST_SERVER_FLAG_PREFIX)
  optimizer_utils.define_lr_schedule_flags(TEST_CLIENT_FLAG_PREFIX)

# Create a list of `(test name, optimizer name flag value, optimizer class)`
# for parameterized tests.
_OPTIMIZERS_TO_TEST = [
    (name, name, cls)
    for name, cls in optimizer_utils._SUPPORTED_OPTIMIZERS.items()
]


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_optimizer_fn_from_flags_invalid_optimizer(self):
    FLAGS['{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX)].value = 'foo'
    with self.assertRaisesRegex(ValueError, 'not a valid optimizer'):
      optimizer_utils.create_optimizer_fn_from_flags(TEST_CLIENT_FLAG_PREFIX)

  def test_create_optimizer_fn_with_no_learning_rate(self):
    with flag_sandbox({
        '{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX): 'sgd',
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX): None
    }):
      with self.assertRaisesRegex(ValueError, 'Learning rate'):
        optimizer_utils.create_optimizer_fn_from_flags(TEST_CLIENT_FLAG_PREFIX)

  def test_create_optimizer_fn_from_flags_flags_set_not_for_optimizer(self):
    with flag_sandbox({'{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX): 'sgd'}):
      # Set an Adam flag that isn't used in SGD.
      # We need to use `_parse_args` because that is the only way FLAGS is
      # notified that a non-default value is being used.
      bad_adam_flag = '{}_adam_beta_1'.format(TEST_CLIENT_FLAG_PREFIX)
      FLAGS._parse_args(
          args=['--{}=0.5'.format(bad_adam_flag)], known_only=True)
      with self.assertRaisesRegex(
          ValueError,
          r'Commandline flags for .*\[sgd\].*\'test_client_adam_beta_1\'.*'):
        optimizer_utils.create_optimizer_fn_from_flags(TEST_CLIENT_FLAG_PREFIX)
      FLAGS[bad_adam_flag].unparse()

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_create_client_optimizer_from_flags(self, optimizer_name,
                                              optimizer_cls):
    commandline_set_learning_rate = 100.0
    with flag_sandbox({
        '{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX):
            optimizer_name,
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX):
            commandline_set_learning_rate
    }):

      custom_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      custom_optimizer = custom_optimizer_fn()
      self.assertIsInstance(custom_optimizer, optimizer_cls)
      self.assertEqual(custom_optimizer.get_config()['learning_rate'],
                       commandline_set_learning_rate)
      custom_optimizer_with_arg = custom_optimizer_fn(11.0)
      self.assertIsInstance(custom_optimizer_with_arg, optimizer_cls)
      self.assertEqual(
          custom_optimizer_with_arg.get_config()['learning_rate'], 11.0)

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_create_server_optimizer_from_flags(self, optimizer_name,
                                              optimizer_cls):
    commandline_set_learning_rate = 100.0
    with flag_sandbox({
        '{}_optimizer'.format(TEST_SERVER_FLAG_PREFIX):
            optimizer_name,
        '{}_learning_rate'.format(TEST_SERVER_FLAG_PREFIX):
            commandline_set_learning_rate
    }):
      custom_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags(
          TEST_SERVER_FLAG_PREFIX)
      custom_optimizer = custom_optimizer_fn()
      self.assertIsInstance(custom_optimizer, optimizer_cls)
      self.assertEqual(custom_optimizer.get_config()['learning_rate'],
                       commandline_set_learning_rate)
      custom_optimizer_with_arg = custom_optimizer_fn(11.0)
      self.assertIsInstance(custom_optimizer_with_arg, optimizer_cls)
      self.assertEqual(custom_optimizer_with_arg.get_config()['learning_rate'],
                       11.0)

  def test_create_constant_client_lr_schedule_from_flags(self):
    with flag_sandbox({
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX): 3.0,
        '{}_lr_schedule'.format(TEST_CLIENT_FLAG_PREFIX): 'constant'
    }):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 3.0, err=1e-5)
      self.assertNear(lr_schedule(1), 3.0, err=1e-5)
      self.assertNear(lr_schedule(105), 3.0, err=1e-5)
      self.assertNear(lr_schedule(1042), 3.0, err=1e-5)

  def test_create_exp_decay_client_lr_schedule_from_flags(self):
    with flag_sandbox({
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX): 3.0,
        '{}_lr_schedule'.format(TEST_CLIENT_FLAG_PREFIX): 'exp_decay',
        '{}_lr_decay_steps'.format(TEST_CLIENT_FLAG_PREFIX): 10,
        '{}_lr_decay_rate'.format(TEST_CLIENT_FLAG_PREFIX): 0.1,
        '{}_lr_staircase'.format(TEST_CLIENT_FLAG_PREFIX): True,
    }):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 3.0, err=1e-5)
      self.assertNear(lr_schedule(3), 3.0, err=1e-5)
      self.assertNear(lr_schedule(10), 0.3, err=1e-5)
      self.assertNear(lr_schedule(19), 0.3, err=1e-5)
      self.assertNear(lr_schedule(20), 0.03, err=1e-5)

    with flag_sandbox({
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX): 3.0,
        '{}_lr_schedule'.format(TEST_CLIENT_FLAG_PREFIX): 'exp_decay',
        '{}_lr_decay_steps'.format(TEST_CLIENT_FLAG_PREFIX): 10,
        '{}_lr_decay_rate'.format(TEST_CLIENT_FLAG_PREFIX): 0.1,
        '{}_lr_staircase'.format(TEST_CLIENT_FLAG_PREFIX): False,
    }):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 3.0, err=1e-5)
      self.assertNear(lr_schedule(1), 2.38298470417, err=1e-5)
      self.assertNear(lr_schedule(10), 0.3, err=1e-5)
      self.assertNear(lr_schedule(25), 0.00948683298, err=1e-5)

  def test_create_inv_lin_client_lr_schedule_from_flags(self):
    with flag_sandbox({
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX): 5.0,
        '{}_lr_schedule'.format(TEST_CLIENT_FLAG_PREFIX): 'inv_lin_decay',
        '{}_lr_decay_steps'.format(TEST_CLIENT_FLAG_PREFIX): 10,
        '{}_lr_decay_rate'.format(TEST_CLIENT_FLAG_PREFIX): 10.0,
        '{}_lr_staircase'.format(TEST_CLIENT_FLAG_PREFIX): True,
    }):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 5.0, err=1e-5)
      self.assertNear(lr_schedule(1), 5.0, err=1e-5)
      self.assertNear(lr_schedule(10), 0.454545454545, err=1e-5)
      self.assertNear(lr_schedule(19), 0.454545454545, err=1e-5)
      self.assertNear(lr_schedule(20), 0.238095238095, err=1e-5)

    with flag_sandbox({
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX): 5.0,
        '{}_lr_schedule'.format(TEST_CLIENT_FLAG_PREFIX): 'inv_lin_decay',
        '{}_lr_decay_steps'.format(TEST_CLIENT_FLAG_PREFIX): 10,
        '{}_lr_decay_rate'.format(TEST_CLIENT_FLAG_PREFIX): 10.0,
        '{}_lr_staircase'.format(TEST_CLIENT_FLAG_PREFIX): False,
    }):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 5.0, err=1e-5)
      self.assertNear(lr_schedule(1), 2.5, err=1e-5)
      self.assertNear(lr_schedule(9), 0.5, err=1e-5)
      self.assertNear(lr_schedule(19), 0.25, err=1e-5)

  def test_create_inv_sqrt_client_lr_schedule_from_flags(self):
    with flag_sandbox({
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX): 2.0,
        '{}_lr_schedule'.format(TEST_CLIENT_FLAG_PREFIX): 'inv_sqrt_decay',
        '{}_lr_decay_steps'.format(TEST_CLIENT_FLAG_PREFIX): 10,
        '{}_lr_decay_rate'.format(TEST_CLIENT_FLAG_PREFIX): 10.0,
        '{}_lr_staircase'.format(TEST_CLIENT_FLAG_PREFIX): True,
    }):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 2.0, err=1e-5)
      self.assertNear(lr_schedule(1), 2.0, err=1e-5)
      self.assertNear(lr_schedule(10), 0.603022689155, err=1e-5)
      self.assertNear(lr_schedule(19), 0.603022689155, err=1e-5)
      self.assertNear(lr_schedule(20), 0.436435780472, err=1e-5)

    with flag_sandbox({
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX): 2.0,
        '{}_lr_schedule'.format(TEST_CLIENT_FLAG_PREFIX): 'inv_sqrt_decay',
        '{}_lr_decay_steps'.format(TEST_CLIENT_FLAG_PREFIX): 10,
        '{}_lr_decay_rate'.format(TEST_CLIENT_FLAG_PREFIX): 10.0,
        '{}_lr_staircase'.format(TEST_CLIENT_FLAG_PREFIX): False,
    }):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 2.0, err=1e-5)
      self.assertNear(lr_schedule(3), 1.0, err=1e-5)
      self.assertNear(lr_schedule(99), 0.2, err=1e-5)
      self.assertNear(lr_schedule(399), 0.1, err=1e-5)

  def test_initial_yogi_accumulator(self):
    x1 = 2 * np.ones([2, 1])
    y1 = np.ones([2, 1])
    x2 = 3 * np.ones([2, 1])
    y2 = np.ones([2, 1])
    tff_dataset = tff.simulation.FromTensorSlicesClientData({
        '1': (x1, y1),
        '2': (x2, y2)
    })
    tff_dataset = tff_dataset.preprocess(lambda x: x.batch(2))
    input_spec = tff_dataset.create_tf_dataset_for_client('1').element_spec

    def model_builder():
      # Create a simple linear regression model, single output.
      return tf.keras.Sequential([
          tf.keras.layers.Dense(
              1,
              kernel_initializer='zeros',
              bias_initializer='zeros',
              input_shape=(1,))
      ])

    tff_model = tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError())
    # (x1, y1) should have gradient [-4, -2]
    # (x2, y2) should have gradient [-6, -2]
    yogi_init = optimizer_utils.compute_yogi_init(
        tff_dataset, tff_model, num_clients=2)
    predicted_yogi_init1 = ((-4.0)**2 + (-2.0)**2) / 2
    predicted_yogi_init2 = ((-6.0)**2 + (-2.0)**2) / 2
    predicted_yogi_init = (predicted_yogi_init1 + predicted_yogi_init2) / 2.0
    self.assertNear(yogi_init, predicted_yogi_init, err=1e-8)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
