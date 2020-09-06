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

import collections
import os.path

import attr
import tensorflow as tf

from tensorflow_federated_research.utils import checkpoint_utils


@attr.s(frozen=True)
class FakeExperimentState(object):
  """An example container for state that need to be stored in the checkpoint.

  Attributes:
    state: A nested structure of `tf.Tensor` instances.
    metrics: A `dict` of `str` keys to `tf.Tensor` values.
    round_num: the iteration number of the current round.
  """
  state = attr.ib()
  metrics = attr.ib()
  round_num = attr.ib()


def build_fake_state():
  return FakeExperimentState(
      state=collections.OrderedDict([
          ('model', {
              'trainable': {
                  'w': tf.constant(1.0),
                  'b': tf.constant(0.0)
              },
              'non_trainable': {}
          }),
          ('optimizer', [tf.Variable(initial_value=1.0, name='learning_rate')]),
      ]),
      metrics={
          'loss': 1.0,
          'accuracy': 0.5
      },
      round_num=1)


class SavedStateTest(tf.test.TestCase):

  def test_save_and_load_roundtrip(self):
    state = build_fake_state()
    export_dir = os.path.join(self.get_temp_dir(), 'ckpt_1')
    checkpoint_utils.save(state, export_dir)

    loaded_state = checkpoint_utils.load(export_dir, state)
    self.assertEqual(state, loaded_state)

  def test_latest_checkpoint_if_path_exists(self):
    self._test_latest_checkpoint(self.get_temp_dir())

  def test_latest_checkpoint_if_path_doesnt_exist(self):
    self._test_latest_checkpoint(os.path.join(self.get_temp_dir(), 'foo'))

  def _test_latest_checkpoint(self, dir_path):
    prefix = 'chkpnt_'
    latest_checkpoint = checkpoint_utils.latest_checkpoint(dir_path, prefix)
    self.assertIsNone(latest_checkpoint)

    # Create checkpoints and ensure that the latest checkpoint found is
    # always the most recently created path.
    state = build_fake_state()
    for round_num in range(5):
      export_dir = os.path.join(dir_path, '{}{:03d}'.format(prefix, round_num))
      checkpoint_utils.save(state, export_dir, prefix)
      latest_checkpoint_path = checkpoint_utils.latest_checkpoint(
          dir_path, prefix)
      self.assertEndsWith(
          latest_checkpoint_path,
          '{:03d}'.format(round_num),
          msg=latest_checkpoint_path)

    # Delete the checkpoints in reverse order and ensure the latest checkpoint
    # decreases.
    for round_num in reversed(range(2, 5)):
      export_dir = os.path.join(dir_path, '{}{:03d}'.format(prefix, round_num))
      tf.io.gfile.rmtree(export_dir)
      latest_checkpoint_path = checkpoint_utils.latest_checkpoint(
          dir_path, prefix)
      self.assertEndsWith(
          latest_checkpoint_path,
          '{}{:03d}'.format(prefix, round_num - 1),
          msg=latest_checkpoint_path)


if __name__ == '__main__':
  tf.test.main()
