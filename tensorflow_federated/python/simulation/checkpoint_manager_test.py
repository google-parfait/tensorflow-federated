# Copyright 2019, Google LLC.
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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.simulation import checkpoint_manager


def _create_test_state(value=0):
  return collections.OrderedDict([
      ('a', {
          'b': tf.constant(value),
          'c': tf.constant(value),
      }),
  ])


class FileCheckpointManagerLoadLatestCheckpointTest(tf.test.TestCase):

  def test_returns_none_with_no_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    structure = _create_test_state()

    state, round_num = checkpoint_mngr.load_latest_checkpoint(structure)

    self.assertIsNone(state)
    self.assertIsNone(round_num)

  def test_returns_state_and_round_num_with_one_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)
    structure = _create_test_state()

    state, round_num = checkpoint_mngr.load_latest_checkpoint(structure)

    self.assertEqual(state, test_state_1)
    self.assertEqual(round_num, 1)

  def test_returns_state_and_round_num_with_three_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)
    test_state_2 = _create_test_state(2)
    checkpoint_mngr.save_checkpoint(test_state_2, 2)
    test_state_3 = _create_test_state(3)
    checkpoint_mngr.save_checkpoint(test_state_3, 3)
    structure = _create_test_state()

    state, round_num = checkpoint_mngr.load_latest_checkpoint(structure)

    self.assertEqual(state, test_state_3)
    self.assertEqual(round_num, 3)

  def test_raises_value_error_with_bad_structure(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)
    structure = None

    with self.assertRaises(ValueError):
      checkpoint_mngr.load_latest_checkpoint(structure)


class FileCheckpointManagerLoadCheckpointTest(tf.test.TestCase):

  def test_returns_state_with_one_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)
    structure = _create_test_state()

    state = checkpoint_mngr.load_checkpoint(structure, 1)

    self.assertEqual(state, test_state_1)

  def test_returns_state_with_three_checkpoint_for_first_round(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)
    test_state_2 = _create_test_state(2)
    checkpoint_mngr.save_checkpoint(test_state_2, 2)
    test_state_3 = _create_test_state(3)
    checkpoint_mngr.save_checkpoint(test_state_3, 3)
    structure = _create_test_state()

    state = checkpoint_mngr.load_checkpoint(structure, 1)

    self.assertEqual(state, test_state_1)

  def test_returns_state_with_three_checkpoint_for_second_round(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)
    test_state_2 = _create_test_state(2)
    checkpoint_mngr.save_checkpoint(test_state_2, 2)
    test_state_3 = _create_test_state(3)
    checkpoint_mngr.save_checkpoint(test_state_3, 3)
    structure = _create_test_state()

    state = checkpoint_mngr.load_checkpoint(structure, 2)

    self.assertEqual(state, test_state_2)

  def test_returns_state_with_three_checkpoint_for_third_round(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)
    test_state_2 = _create_test_state(2)
    checkpoint_mngr.save_checkpoint(test_state_2, 2)
    test_state_3 = _create_test_state(3)
    checkpoint_mngr.save_checkpoint(test_state_3, 3)
    structure = _create_test_state()

    state = checkpoint_mngr.load_checkpoint(structure, 3)

    self.assertEqual(state, test_state_3)

  def test_raises_file_not_found_error_with_no_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    structure = _create_test_state()

    with self.assertRaises(FileNotFoundError):
      _ = checkpoint_mngr.load_checkpoint(structure, 0)

  def test_raises_file_not_found_error_with_one_checkpoint_for_bad_round(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)
    structure = _create_test_state()

    with self.assertRaises(FileNotFoundError):
      _ = checkpoint_mngr.load_checkpoint(structure, 10)

  def test_raises_value_error_with_bad_structure(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)
    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)
    structure = None

    with self.assertRaises(ValueError):
      checkpoint_mngr.load_checkpoint(structure, 1)


class FileCheckpointManagerSaveCheckpointTest(tf.test.TestCase,
                                              parameterized.TestCase):

  def test_saves_one_checkpoint(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)

    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_1'])

  def test_saves_three_checkpoints(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)

    for i in range(1, 4):
      test_state = _create_test_state(i)
      checkpoint_mngr.save_checkpoint(test_state, i)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_1', 'ckpt_2', 'ckpt_3'])

  def test_removes_oldest_with_keep_first_true(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(
        temp_dir, keep_total=3, keep_first=True)

    for i in range(1, 5):
      test_state = _create_test_state(i)
      checkpoint_mngr.save_checkpoint(test_state, i)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_1', 'ckpt_3', 'ckpt_4'])

  def test_removes_oldest_with_keep_first_false(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(
        temp_dir, keep_total=3, keep_first=False)

    for i in range(1, 5):
      test_state = _create_test_state(i)
      checkpoint_mngr.save_checkpoint(test_state, i)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_2', 'ckpt_3', 'ckpt_4'])

  @parameterized.named_parameters(
      ('keep_total_equal_to_zero', 0),
      ('keep_total_smaller_than_zero', -1),
  )
  def test_keep_all_checkpoints(self, keep_total):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(
        temp_dir, keep_total=keep_total, keep_first=False)

    for i in range(1, 4):
      test_state = _create_test_state(i)
      checkpoint_mngr.save_checkpoint(test_state, i)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_1', 'ckpt_2', 'ckpt_3'])

  def test_raises_already_exists_error_with_existing_round_number(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(temp_dir)

    test_state_1 = _create_test_state(1)
    checkpoint_mngr.save_checkpoint(test_state_1, 1)

    with self.assertRaises(tf.errors.AlreadyExistsError):
      checkpoint_mngr.save_checkpoint(test_state_1, 1)

  def test_save_with_nondefault_checkpoints_per_round(self):
    temp_dir = self.get_temp_dir()
    checkpoint_mngr = checkpoint_manager.FileCheckpointManager(
        temp_dir, step=3, keep_total=3)

    for i in range(8):
      test_state = _create_test_state(i)
      checkpoint_mngr.save_checkpoint(test_state, i)

    self.assertCountEqual(os.listdir(temp_dir), ['ckpt_0', 'ckpt_3', 'ckpt_6'])


if __name__ == '__main__':
  tf.test.main()
