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

import collections
import os.path

import tensorflow as tf

from tensorflow_federated.python.research.utils import checkpoint_manager

tf.compat.v1.enable_v2_behavior()


def _create_dummy_structure():
  return collections.OrderedDict([
      ('a', {
          'b': tf.constant(0.0),
          'c': tf.constant(0.0),
      }),
  ])


def _create_dummy_state():
  return collections.OrderedDict([
      ('a', {
          'b': tf.constant(1.0),
          'c': tf.constant(1.0),
      }),
  ])


class FileCheckpointManagerLoadLatestCheckpointOrDefaultTest(tf.test.TestCase):

  def test_returns_default_and_zero_with_no_checkpoints_and_also_saves(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    default = _create_dummy_structure()

    state, round_num = checkpoint_mngr.load_latest_checkpoint_or_default(
        default)

    self.assertEqual(state, default)
    self.assertEqual(round_num, 0)

    self.assertEqual(set(os.listdir(temp_dir)), set(['ckpt_0']))


class FileCheckpointManagerLoadLatestCheckpointTest(tf.test.TestCase):

  def test_returns_none_and_zero_with_no_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    structure = _create_dummy_structure()

    state, round_num = checkpoint_mngr.load_latest_checkpoint(structure)

    self.assertIsNone(state)
    self.assertEqual(round_num, 0)

  def test_returns_state_and_round_num_with_one_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    state = _create_dummy_state()
    checkpoint_mngr.save_checkpoint(state, 1)
    structure = _create_dummy_structure()

    state, round_num = checkpoint_mngr.load_latest_checkpoint(structure)

    expected_state = _create_dummy_state()
    self.assertEqual(state, expected_state)
    self.assertEqual(round_num, 1)

  def test_returns_state_and_round_num_with_three_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    state = _create_dummy_state()
    checkpoint_mngr.save_checkpoint(state, 1)
    checkpoint_mngr.save_checkpoint(state, 2)
    checkpoint_mngr.save_checkpoint(state, 3)
    structure = _create_dummy_structure()

    state, round_num = checkpoint_mngr.load_latest_checkpoint(structure)

    expected_state = _create_dummy_state()
    self.assertEqual(state, expected_state)
    self.assertEqual(round_num, 3)

  def test_raises_value_error_with_bad_structure(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    state = _create_dummy_state()
    checkpoint_mngr.save_checkpoint(state, 1)
    structure = None

    with self.assertRaises(ValueError):
      _, _ = checkpoint_mngr.load_latest_checkpoint(structure)


class FileCheckpointManagerLoadCheckpointTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._checkpoint_mngr = checkpoint_manager.FileCheckpointManager(
        self.get_temp_dir())

  def test_raises_file_not_found_error_for_no_checkpoint(self):
    structure = _create_dummy_structure()

    with self.assertRaises(FileNotFoundError):
      _ = self._checkpoint_mngr.load_checkpoint(structure, 0)

  def test_returns_state_for_checkpoint(self):
    state = _create_dummy_state()
    round_num = 1
    self._checkpoint_mngr.save_checkpoint(state, round_num)
    structure = _create_dummy_structure()

    loaded_state = self._checkpoint_mngr.load_checkpoint(structure, round_num)

    self.assertEqual(loaded_state, state)

  def test_raises_value_error_with_bad_structure(self):
    state = _create_dummy_state()
    round_num = 1
    self._checkpoint_mngr.save_checkpoint(state, round_num)
    structure = None

    with self.assertRaises(ValueError):
      _, _ = self._checkpoint_mngr.load_checkpoint(structure, round_num)


class FileCheckpointManagerSaveCheckpointTest(tf.test.TestCase):

  def test_saves_one_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    state = _create_dummy_state()

    checkpoint_mngr.save_checkpoint(state, 1)

    self.assertEqual(set(os.listdir(temp_dir)), set(['ckpt_1']))

  def test_saves_three_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    state = _create_dummy_state()

    checkpoint_mngr.save_checkpoint(state, 1)
    checkpoint_mngr.save_checkpoint(state, 2)
    checkpoint_mngr.save_checkpoint(state, 3)

    self.assertEqual(
        set(os.listdir(temp_dir)), set(['ckpt_1', 'ckpt_2', 'ckpt_3']))

  def test_removes_oldest_with_keep_first_true(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(
        temp_dir, keep_total=3, keep_first=True)
    state = _create_dummy_state()

    checkpoint_mngr.save_checkpoint(state, 1)
    checkpoint_mngr.save_checkpoint(state, 2)
    checkpoint_mngr.save_checkpoint(state, 3)
    checkpoint_mngr.save_checkpoint(state, 4)

    self.assertEqual(
        set(os.listdir(temp_dir)), set(['ckpt_1', 'ckpt_3', 'ckpt_4']))

  def test_removes_oldest_with_keep_first_false(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(
        temp_dir, keep_total=3, keep_first=False)
    state = _create_dummy_state()

    checkpoint_mngr.save_checkpoint(state, 1)
    checkpoint_mngr.save_checkpoint(state, 2)
    checkpoint_mngr.save_checkpoint(state, 3)
    checkpoint_mngr.save_checkpoint(state, 4)

    self.assertEqual(
        set(os.listdir(temp_dir)), set(['ckpt_2', 'ckpt_3', 'ckpt_4']))

  def test_raises_already_exists_error_with_existing_round_number(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    state = _create_dummy_state()

    checkpoint_mngr.save_checkpoint(state, 1)

    with self.assertRaises(tf.errors.AlreadyExistsError):
      checkpoint_mngr.save_checkpoint(state, 1)


if __name__ == '__main__':
  tf.test.main()
