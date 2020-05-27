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
"""Save or load a nested structure."""

import os.path
import re

from absl import logging
import tensorflow as tf


def get_serial_number(export_dir, prefix='ckpt_'):
  r"""Get the integer component of a checkpoint directory name.

  Args:
    export_dir: A checkpoint directory.
    prefix: Common prefix shared by all checkpoint directories.

  Returns:
    The number extracted from the checkpoint directory, or -1 if the directory
    is not formatted correctly.
  """
  matcher = re.match(r'^{}(?P<num>\d+)$'.format(prefix),
                     os.path.basename(export_dir))
  return int(matcher.group('num')) if matcher else -1


def latest_checkpoint(root_output_dir, prefix='ckpt_'):
  r"""Get the latest checkpoint name.

  Searches `root_output_dir` for directories matching the regular expression
  `prefix_\d+$` and returns the directory with the largest integer suffix.

  Args:
    root_output_dir: The directory where all checkpoints stored.
    prefix: The common prefix shared by all checkpoint directories.

  Returns:
    Dirname of the latest checkpoint. If there are no checkpoints (or
    root_output_dir does not exist), returns None.
  """
  if not tf.io.gfile.exists(root_output_dir):
    return None
  checkpoints = tf.io.gfile.glob(
      os.path.join(root_output_dir, '{}*'.format(prefix)))
  if not checkpoints:
    return None
  return max(checkpoints, key=lambda ckpt: get_serial_number(ckpt, prefix))


def save(obj, export_dir, prefix=None):
  r"""Save a nested structure to `export_dir`.

  Note: to be compatible with `latest_checkpoint`, the basename of `export_dir`
  must follow the regular expression pattern `<prefix>\d+`, where the final
  digit matcher  determines the ordering of the checkpoints.

  Args:
    obj: A nested structure which `tf.convert_to_tensor` supports.
    export_dir: A directory in which to write the state.
    prefix: The common prefix shared by all checkpoint directories. If provided,
      we will fail if the export directory doesn't match this prefix. If not
      provided, no check will be performed.

  Raises:
    ValueError: If `prefix` is provided and `export_dir` doesn't use the prefix.
  """
  if prefix is not None and get_serial_number(export_dir, prefix) < 0:
    raise ValueError('Checkpoint dir "{}" is not named like "{}XXXX!'.format(
        export_dir, prefix))

  model = tf.Module()
  model.obj = tf.nest.flatten(obj)
  model.build_obj_fn = tf.function(lambda: model.obj, input_signature=())

  # First write to a temporary directory.
  temp_export_dir = os.path.join(
      os.path.dirname(export_dir), '.temp_' + os.path.basename(export_dir))
  try:
    tf.io.gfile.rmtree(temp_export_dir)
  except tf.errors.NotFoundError:
    pass
  tf.io.gfile.makedirs(temp_export_dir)
  tf.saved_model.save(model, temp_export_dir, signatures={})

  # Rename the temp directory to the final location atomically.
  tf.io.gfile.rename(temp_export_dir, export_dir)
  logging.info('Checkpoint saved to: %s', export_dir)


def load(export_dir, obj_template):
  """Load a nested structure from `export_dir`.

  Args:
    export_dir: The directory to load from.
    obj_template: An object that provides the nested structure to mimic.

  Returns:
    Loaded nested structure.

  Raises:
    FileNotFoundError: No such file or directory.
  """
  if tf.io.gfile.exists(export_dir):
    loaded = tf.compat.v2.saved_model.load(export_dir)

    flat_obj = loaded.build_obj_fn()
    obj = tf.nest.pack_sequence_as(obj_template, flat_obj)

    logging.info('Checkpoint loaded from: %s', export_dir)
  else:
    raise FileNotFoundError('No such file or directory: %s' % export_dir)

  return obj
