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
"""Utilities for saving and loading experiment checkpoints."""

import os.path
import re
from typing import Any, List, Tuple, Union

from absl import logging
import tensorflow as tf


class FileCheckpointManager():
  """A checkpoint manager backed by a file system.

  This checkpoint manager is a utility to save and load checkpoints. While
  the checkpoint manager is compatible with any nested structure supported by
  `tf.convert_to_tensor`, checkpoints may often represent the output of a
  `tff.templates.IterativeProcess`. For example, one possible use case would
  be to save the `ServerState` output of an iterative process created via
  `tff.learning`. This is comparable to periodically saving model weights and
  optimizer states during non-federated training.

  The implementation you find here is slightly different from
  `tf.train.CheckpointManager`. This implementation yields nested structures
  that are immutable whereas `tf.train.CheckpointManager` is used to manage
  `tf.train.Checkpoint` objects, which are mutable collections. Additionally,
  this implementation allows retaining the initial checkpoint as part of the
  total number of checkpoints that are kept.

  The checkpoint manager is intended only for allowing simulations to be
  resumed after interruption. In particular, it is intended to only restart the
  same simulation, run with the same version of TensorFlow Federated.
  """

  def __init__(self,
               root_dir: str,
               prefix: str = 'ckpt_',
               keep_total: int = 5,
               keep_first: bool = True):
    """Returns an initialized `FileCheckpointManager`.

    Args:
      root_dir: A path on the filesystem to store checkpoints.
      prefix: A string to use as the prefix for checkpoint names.
      keep_total: An integer representing the total number of checkpoints to
        keep.
      keep_first: A boolean indicating if the first checkpoint should be kept,
        irrespective of whether it is in the last `keep_total` checkpoints. This
        is desirable in settings where you would like to ensure full
        reproducibility of the simulation, especially in settings where
        model weights or optimizer states are initialized randomly. By loading
        from the initial checkpoint, one can avoid re-initializing and obtaining
        different results.
    """
    super().__init__()
    self._root_dir = root_dir
    self._prefix = prefix
    self._keep_total = keep_total
    self._keep_first = keep_first
    path = re.escape(os.path.join(root_dir, prefix))
    self._round_num_expression = re.compile(r'{}([0-9]+)$'.format(path))

  def load_latest_checkpoint_or_default(self, default: Any) -> Tuple[Any, int]:
    """Loads latest checkpoint, loading `default` if no checkpoints exist.

    Saves `default` as the 0th checkpoint if no checkpoints exist.

    Args:
      default: A nested structure which `tf.convert_to_tensor` supports to use
        as a template when reconstructing the loaded template. This structure
        will be saved as the checkpoint for round number 0 and returned if there
        are no pre-existing saved checkpoints.

    Returns:
      A `tuple` of `(state, round_num)` where `state` matches the Python
      structure in `structure`, and `round_num` is an integer. If no
      checkpoints have been written, returns `(default, 0)`.
    """
    state, round_num = self.load_latest_checkpoint(default)
    if state is None:
      state = default
      round_num = 0
      self.save_checkpoint(state, round_num)
    return state, round_num

  def load_latest_checkpoint(self,
                             structure: Any) -> Tuple[Any, Union[int, None]]:
    """Loads the latest state and round number.

    Args:
      structure: A nested structure which `tf.convert_to_tensor` supports to use
        as a template when reconstructing the loaded template.

    Returns:
      A `tuple` of `(state, round_num)` where `state` matches the Python
      structure in `structure`, and `round_num` is an integer. If no checkpoints
      have been previously saved, returns the tuple `(None, None)`.
    """
    checkpoint_paths = self._get_all_checkpoint_paths()
    if checkpoint_paths:
      checkpoint_path = max(checkpoint_paths, key=self._round_num)
      return self._load_checkpoint_from_path(structure, checkpoint_path)
    return None, None

  def load_checkpoint(self, structure: Any, round_num: int) -> Any:
    """Returns the checkpointed state for the given `round_num`.

    Args:
      structure: A nested structure which `tf.convert_to_tensor` supports to use
        as a template when reconstructing the loaded template.
      round_num: An integer representing the round to load from.
    """
    basename = '{}{}'.format(self._prefix, round_num)
    checkpoint_path = os.path.join(self._root_dir, basename)
    state, _ = self._load_checkpoint_from_path(structure, checkpoint_path)
    return state

  def _load_checkpoint_from_path(self, structure: Any,
                                 checkpoint_path: str) -> Tuple[Any, int]:
    """Returns the state and round number for the given `checkpoint_path`.

    Args:
      structure: A nested structure which `tf.convert_to_tensor` supports to use
        as a template when reconstructing the loaded template.
      checkpoint_path: A path on the filesystem to load.

    Raises:
      FileNotFoundError: If a checkpoint for given `checkpoint_path` doesn't
        exist.
    """
    if not tf.io.gfile.exists(checkpoint_path):
      raise FileNotFoundError(
          'No such file or directory: {}'.format(checkpoint_path))
    model = tf.saved_model.load(checkpoint_path)
    flat_obj = model.build_obj_fn()
    state = tf.nest.pack_sequence_as(structure, flat_obj)
    round_num = self._round_num(checkpoint_path)
    logging.info('Checkpoint loaded: %s', checkpoint_path)
    return state, round_num

  def save_checkpoint(self, state: Any, round_num: int) -> None:
    """Saves a new checkpointed `state` for the given `round_num`.

    Args:
      state: A nested structure which `tf.convert_to_tensor` supports.
      round_num: An integer representing the current training round.
    """
    basename = '{}{}'.format(self._prefix, round_num)
    checkpoint_path = os.path.join(self._root_dir, basename)
    flat_obj = tf.nest.flatten(state)
    model = tf.Module()
    model.obj = flat_obj
    model.build_obj_fn = tf.function(lambda: model.obj, input_signature=())

    # First write to a temporary directory.
    temp_basename = '.temp_{}'.format(basename)
    temp_path = os.path.join(self._root_dir, temp_basename)
    try:
      tf.io.gfile.rmtree(temp_path)
    except tf.errors.NotFoundError:
      pass
    tf.io.gfile.makedirs(temp_path)
    tf.saved_model.save(model, temp_path, signatures={})

    # Rename the temp directory to the final location atomically.
    tf.io.gfile.rename(temp_path, checkpoint_path)
    logging.info('Checkpoint saved: %s', checkpoint_path)

    self._clear_old_checkpoints()

  def _clear_old_checkpoints(self) -> None:
    """Removes old checkpoints."""
    checkpoint_paths = self._get_all_checkpoint_paths()
    if len(checkpoint_paths) > self._keep_total:
      checkpoint_paths = sorted(checkpoint_paths, key=self._round_num)
      start = 1 if self._keep_first else 0
      stop = start - self._keep_total
      for checkpoint_path in checkpoint_paths[start:stop]:
        tf.io.gfile.rmtree(checkpoint_path)
        logging.info('Checkpoint removed: %s', checkpoint_path)

  def _round_num(self, checkpoint_path: str) -> int:
    """Returns the round number for the given `checkpoint_path`, or `-1`."""
    match = self._round_num_expression.match(checkpoint_path)
    if match is None:
      logging.debug(
          'Could not extract round number from: \'%s\' using the following '
          'pattern: \'%s\'', checkpoint_path,
          self._round_num_expression.pattern)
      return -1
    return int(match.group(1))

  def _get_all_checkpoint_paths(self) -> List[str]:
    """Returns all the checkpoint paths managed by the instance."""
    # Due to tensorflow/issues/19378, we cannot use `tf.io.gfile.glob` here
    # because it returns directory contents recursively on Windows.
    if tf.io.gfile.exists(self._root_dir):
      root_dir_entries = tf.io.gfile.listdir(self._root_dir)
      return [
          os.path.join(self._root_dir, e)
          for e in root_dir_entries
          if e.startswith(self._prefix)
      ]
    else:
      return []
