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
"""Tests for utils."""

import os

from absl import flags
from absl.testing import absltest
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow_federated.python.research.utils import utils_impl as utils

FLAGS = flags.FLAGS


class UtilsTest(tf.test.TestCase):

  def test_get_optimizer_from_flags(self):
    utils.define_optimizer_flags('server', defaults=dict(learning_rate=1.25))
    self.assertEqual(FLAGS.server_learning_rate, 1.25)
    optimizer = utils.get_optimizer_from_flags('server')
    self.assertEqual(optimizer.get_config()['learning_rate'], 1.25)

  def test_define_optimizer_unused_default(self):
    with self.assertRaisesRegex(ValueError, 'not consumed'):
      # Use a different prefix to avoid declaring duplicate flags:
      utils.define_optimizer_flags('client', defaults=dict(lr=1.25))

  def test_atomic_write(self):
    # Ensure randomness for temp filenames.
    np.random.seed()

    for name in ['foo.csv', 'baz.csv.bz2']:
      dataframe = pd.DataFrame(dict(a=[1, 2], b=[4.0, 5.0]))
      output_file = os.path.join(absltest.get_default_test_tmpdir(), name)
      utils.atomic_write_to_csv(dataframe, output_file)
      dataframe2 = pd.read_csv(output_file, index_col=0)
      pd.testing.assert_frame_equal(dataframe, dataframe2)

      # Overwriting
      dataframe3 = pd.DataFrame(dict(a=[1, 2, 3], b=[4.0, 5.0, 6.0]))
      utils.atomic_write_to_csv(dataframe3, output_file)
      dataframe4 = pd.read_csv(output_file, index_col=0)
      pd.testing.assert_frame_equal(dataframe3, dataframe4)

  def test_iter_grid(self):
    grid = dict(a=[], b=[])
    self.assertCountEqual(list(utils.iter_grid(grid)), [])

    grid = dict(a=[1])
    self.assertCountEqual(list(utils.iter_grid(grid)), [dict(a=1)])

    grid = dict(a=[1, 2])
    self.assertCountEqual(list(utils.iter_grid(grid)), [dict(a=1), dict(a=2)])

    grid = dict(a=[1, 2], b='b', c=[3.0, 4.0])
    self.assertCountEqual(
        list(utils.iter_grid(grid)), [
            dict(a=1, b='b', c=3.0),
            dict(a=1, b='b', c=4.0),
            dict(a=2, b='b', c=3.0),
            dict(a=2, b='b', c=4.0)
        ])


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
