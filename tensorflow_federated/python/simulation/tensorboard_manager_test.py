# Copyright 2020, Google LLC.
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
import os.path

import tensorflow as tf

from tensorflow_federated.python.simulation import tensorboard_manager


def _create_scalar_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': 1.0,
          'c': 2.0,
      }),
  ])


def _create_nonscalar_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': tf.ones([1]),
          'c': tf.zeros([2, 2]),
      }),
  ])


def _create_scalar_metrics_with_extra_column():
  metrics = _create_scalar_metrics()
  metrics['a']['d'] = 3.0
  return metrics


class TensorBoardManagerTest(tf.test.TestCase):

  def test_scalar_metrics_are_written(self):
    summary_dir = os.path.join(self.get_temp_dir(), 'logdir')
    tb_mngr = tensorboard_manager.TensorBoardManager(summary_dir=summary_dir)
    tb_mngr.save_metrics(_create_scalar_metrics(), 0)
    self.assertTrue(tf.io.gfile.exists(summary_dir))
    self.assertLen(tf.io.gfile.listdir(summary_dir), 1)

  def test_nonscalar_metrics_are_written(self):
    summary_dir = os.path.join(self.get_temp_dir(), 'logdir')
    tb_mngr = tensorboard_manager.TensorBoardManager(summary_dir=summary_dir)
    tb_mngr.save_metrics(_create_nonscalar_metrics(), 0)
    self.assertTrue(tf.io.gfile.exists(summary_dir))
    self.assertLen(tf.io.gfile.listdir(summary_dir), 1)

  def test_flatten_returns_flat_dict(self):
    input_data_dict = _create_scalar_metrics()
    flat_data = tensorboard_manager._flatten_nested_dict(input_data_dict)
    self.assertEqual({
        'a/b': 1.0,
        'a/c': 2.0,
    }, flat_data)

  def test_save_metrics_raises_value_error_if_round_num_is_negative(self):
    tb_mngr = tensorboard_manager.TensorBoardManager(
        summary_dir=self.get_temp_dir())

    with self.assertRaises(ValueError):
      tb_mngr.save_metrics(_create_scalar_metrics(), -1)

  def test_save_metrics_raises_value_error_if_round_num_is_out_of_order(self):
    tb_mngr = tensorboard_manager.TensorBoardManager(
        summary_dir=self.get_temp_dir())

    tb_mngr.save_metrics(_create_scalar_metrics(), 1)

    with self.assertRaises(ValueError):
      tb_mngr.save_metrics(_create_scalar_metrics(), 0)


if __name__ == '__main__':
  tf.test.main()
