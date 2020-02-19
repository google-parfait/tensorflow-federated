# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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

import os.path

import pandas as pd
import tensorflow as tf

from tensorflow_federated.python.research.optimization.shared import keras_callbacks

tf.compat.v1.enable_v2_behavior()


class KerasCallbacksTest(tf.test.TestCase):

  def test_initializes(self):
    tmpdir = self.get_temp_dir()
    logger = keras_callbacks.AtomicCSVLogger(tmpdir)
    self.assertIsInstance(logger, tf.keras.callbacks.Callback)

  def test_writes_dict_as_csv(self):
    tmpdir = self.get_temp_dir()
    logger = keras_callbacks.AtomicCSVLogger(tmpdir)
    logger.on_epoch_end(epoch=0, logs={'value': 0, 'value_1': 'a'})
    logger.on_epoch_end(epoch=1, logs={'value': 2, 'value_1': 'b'})
    logger.on_epoch_end(epoch=2, logs={'value': 3, 'value_1': 'c'})
    logger.on_epoch_end(epoch=1, logs={'value': 4, 'value_1': 'd'})
    read_logs = pd.read_csv(
        os.path.join(tmpdir, 'metric_results.csv'),
        index_col=0,
        header=0,
        engine='c')
    self.assertNotEmpty(read_logs)
    pd.testing.assert_frame_equal(
        read_logs, pd.DataFrame({
            'value': [0, 4],
            'value_1': ['a', 'd'],
        }))


if __name__ == '__main__':
  tf.test.main()
