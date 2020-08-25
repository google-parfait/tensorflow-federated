# Copyright 2020, The TensorFlow Federated Authors.
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

from absl import logging

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.adaptive_lr_decay import callbacks


class ReduceLROnPlateauTest(tf.test.TestCase):

  def test_raises_bad_decay_factor(self):
    with self.assertRaises(ValueError):
      callbacks.create_reduce_lr_on_plateau(
          learning_rate=0.1, decay_factor=2.0, cooldown=0)
    with self.assertRaises(ValueError):
      callbacks.create_reduce_lr_on_plateau(
          learning_rate=0.1, decay_factor=-1.0)

  def test_lr_decay_after_patience_rounds(self):
    lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=1.0,
        decay_factor=0.5,
        minimize=False,
        window_size=3,
        patience=5,
        cooldown=0)
    logging.info('LR Callback: %s', lr_callback)
    self.assertEqual(lr_callback.metrics_window, [0.0, 0.0, 0.0])
    for i in range(4):
      lr_callback = lr_callback.update(-1.0)
      logging.info('LR Callback: %s', lr_callback)
      self.assertEqual(lr_callback.best, 0.0)
      self.assertEqual(lr_callback.learning_rate, 1.0)
      self.assertEqual(lr_callback.wait, i + 1)

    lr_callback = lr_callback.update(-1.0)
    logging.info('LR Callback: %s', lr_callback)
    self.assertEqual(lr_callback.best, 0.0)
    self.assertEqual(lr_callback.learning_rate, 0.5)
    self.assertEqual(lr_callback.wait, 0)

  def test_window_with_inf_values(self):
    lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=1.0,
        decay_factor=0.5,
        minimize=True,
        window_size=3,
        patience=1,
        cooldown=0)
    logging.info('LR Callback: %s', lr_callback)
    self.assertEqual(lr_callback.metrics_window, [np.Inf for _ in range(3)])
    for i in range(2):
      lr_callback = lr_callback.update(3.0)
      logging.info('LR Callback: %s', lr_callback)
      self.assertEqual(lr_callback.best, np.Inf)
      self.assertEqual(lr_callback.learning_rate, (0.5)**(i + 1))
      self.assertEqual(lr_callback.wait, 0)

    lr_callback = lr_callback.update(6.0)
    logging.info('LR Callback: %s', lr_callback)
    self.assertEqual(lr_callback.best, 4.0)
    self.assertEqual(lr_callback.learning_rate, 0.25)
    self.assertEqual(lr_callback.wait, 0)

  def test_min_lr(self):
    lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        decay_factor=0.5,
        min_lr=0.2,
        minimize=False,
        window_size=1,
        patience=1,
        cooldown=0)
    logging.info('LR Callback: %s', lr_callback)
    self.assertEqual(lr_callback.learning_rate, 0.2)
    for i in range(5):
      x = -float(i)
      lr_callback = lr_callback.update(x)
      logging.info('LR Callback: %s', lr_callback)
      self.assertEqual(lr_callback.best, 0.0)
      self.assertEqual(lr_callback.learning_rate, 0.2)
      self.assertEqual(lr_callback.wait, i + 1)

  def test_cooldown(self):
    lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=2.0,
        decay_factor=0.5,
        minimize=False,
        window_size=1,
        patience=0,
        cooldown=3)
    logging.info('LR Callback: %s', lr_callback)
    self.assertEqual(lr_callback.learning_rate, 2.0)
    self.assertEqual(lr_callback.cooldown, 3)
    self.assertEqual(lr_callback.cooldown_counter, 3)
    for i in range(2):
      lr_callback = lr_callback.update(-1.0)
      logging.info('LR Callback: %s', lr_callback)
      self.assertEqual(lr_callback.learning_rate, 2.0)
      self.assertEqual(lr_callback.wait, 0)
      self.assertEqual(lr_callback.cooldown, 3)
      self.assertEqual(lr_callback.cooldown_counter, 2 - i)
    lr_callback = lr_callback.update(-1.0)
    logging.info('LR Callback: %s', lr_callback)
    self.assertEqual(lr_callback.learning_rate, 1.0)
    self.assertEqual(lr_callback.wait, 0)
    self.assertEqual(lr_callback.cooldown, 3)
    self.assertEqual(lr_callback.cooldown_counter, 3)


if __name__ == '__main__':
  tf.test.main()
