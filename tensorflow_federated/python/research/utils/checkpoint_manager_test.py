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
import os
import os.path

import tensorflow as tf

from tensorflow_federated.python.research.utils import checkpoint_manager

tf.compat.v1.enable_v2_behavior()


def _create_dummy_state(value=0):
  return collections.OrderedDict([
      ('a', {
          'b': tf.constant(value),
          'c': tf.constant(value),
      }),
  ])


class FileCheckpointManagerLoadLatestCheckpointOrDefaultTest(tf.test.TestCase):

  def test_saves_and_returns_structure_and_zero_with_no_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    structure = _create_dummy_state()

    state, round_num = checkpoint_mngr.load_latest_checkpoint_or_default(
        structure)

    self.assertEqual(state, structure)
    self.assertEqual(round_num, 0)
    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_0'])


class FileCheckpointManagerLoadLatestCheckpointTest(tf.test.TestCase):

  def test_returns_none_and_zero_with_no_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    structure = _create_dummy_state()

    state, round_num = checkpoint_mngr.load_latest_checkpoint(structure)

    self.assertIsNone(state)
    self.assertEqual(round_num, 0)

  def test_returns_state_and_round_num_with_one_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    structure = _create_dummy_state()

    state, round_num = checkpoint_mngr.load_latest_checkpoint(structure)

    self.assertEqual(state, dummy_state_1)
    self.assertEqual(round_num, 1)

  def test_returns_state_and_round_num_with_three_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    dummy_state_2 = _create_dummy_state(2)
    checkpoint_mngr.save_checkpoint(dummy_state_2, 2)
    dummy_state_3 = _create_dummy_state(3)
    checkpoint_mngr.save_checkpoint(dummy_state_3, 3)
    structure = _create_dummy_state()

    state, round_num = checkpoint_mngr.load_latest_checkpoint(structure)

    self.assertEqual(state, dummy_state_3)
    self.assertEqual(round_num, 3)

  def test_raises_value_error_with_bad_structure(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    structure = None

    with self.assertRaises(ValueError):
      checkpoint_mngr.load_latest_checkpoint(structure)


class FileCheckpointManagerLoadCheckpointTest(tf.test.TestCase):

  def test_returns_state_with_one_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    structure = _create_dummy_state()

    state = checkpoint_mngr.load_checkpoint(structure, 1)

    self.assertEqual(state, dummy_state_1)

  def test_returns_state_with_three_checkpoint_for_first_round(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    dummy_state_2 = _create_dummy_state(2)
    checkpoint_mngr.save_checkpoint(dummy_state_2, 2)
    dummy_state_3 = _create_dummy_state(3)
    checkpoint_mngr.save_checkpoint(dummy_state_3, 3)
    structure = _create_dummy_state()

    state = checkpoint_mngr.load_checkpoint(structure, 1)

    self.assertEqual(state, dummy_state_1)

  def test_returns_state_with_three_checkpoint_for_second_round(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    dummy_state_2 = _create_dummy_state(2)
    checkpoint_mngr.save_checkpoint(dummy_state_2, 2)
    dummy_state_3 = _create_dummy_state(3)
    checkpoint_mngr.save_checkpoint(dummy_state_3, 3)
    structure = _create_dummy_state()

    state = checkpoint_mngr.load_checkpoint(structure, 2)

    self.assertEqual(state, dummy_state_2)

  def test_returns_state_with_three_checkpoint_for_third_round(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    dummy_state_2 = _create_dummy_state(2)
    checkpoint_mngr.save_checkpoint(dummy_state_2, 2)
    dummy_state_3 = _create_dummy_state(3)
    checkpoint_mngr.save_checkpoint(dummy_state_3, 3)
    structure = _create_dummy_state()

    state = checkpoint_mngr.load_checkpoint(structure, 3)

    self.assertEqual(state, dummy_state_3)

  def test_raises_file_not_found_error_with_no_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    structure = _create_dummy_state()

    with self.assertRaises(FileNotFoundError):
      _ = checkpoint_mngr.load_checkpoint(structure, 0)

  def test_raises_file_not_found_error_with_one_checkpoint_for_bad_round(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    structure = _create_dummy_state()

    with self.assertRaises(FileNotFoundError):
      _ = checkpoint_mngr.load_checkpoint(structure, 10)

  def test_raises_value_error_with_bad_structure(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    structure = None

    with self.assertRaises(ValueError):
      checkpoint_mngr.load_checkpoint(structure, 1)


class FileCheckpointManagerSaveCheckpointTest(tf.test.TestCase):

  def test_saves_one_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)

    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_1'])

  def test_saves_three_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)

    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    dummy_state_2 = _create_dummy_state(2)
    checkpoint_mngr.save_checkpoint(dummy_state_2, 2)
    dummy_state_3 = _create_dummy_state(3)
    checkpoint_mngr.save_checkpoint(dummy_state_3, 3)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_1', 'ckpt_2', 'ckpt_3'])

  def test_removes_oldest_with_keep_first_true(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(
        temp_dir, keep_total=3, keep_first=True)

    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    dummy_state_2 = _create_dummy_state(2)
    checkpoint_mngr.save_checkpoint(dummy_state_2, 2)
    dummy_state_3 = _create_dummy_state(3)
    checkpoint_mngr.save_checkpoint(dummy_state_3, 3)
    dummy_state_4 = _create_dummy_state(4)
    checkpoint_mngr.save_checkpoint(dummy_state_4, 4)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_1', 'ckpt_3', 'ckpt_4'])

  def test_removes_oldest_with_keep_first_false(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(
        temp_dir, keep_total=3, keep_first=False)

    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)
    dummy_state_2 = _create_dummy_state(2)
    checkpoint_mngr.save_checkpoint(dummy_state_2, 2)
    dummy_state_3 = _create_dummy_state(3)
    checkpoint_mngr.save_checkpoint(dummy_state_3, 3)
    dummy_state_4 = _create_dummy_state(4)
    checkpoint_mngr.save_checkpoint(dummy_state_4, 4)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_2', 'ckpt_3', 'ckpt_4'])

  def test_raises_already_exists_error_with_existing_round_number(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)

    dummy_state_1 = _create_dummy_state(1)
    checkpoint_mngr.save_checkpoint(dummy_state_1, 1)

    with self.assertRaises(tf.errors.AlreadyExistsError):
      checkpoint_mngr.save_checkpoint(dummy_state_1, 1)


if __name__ == '__main__':
  tf.test.main()
