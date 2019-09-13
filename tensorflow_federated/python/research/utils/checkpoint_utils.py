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
"""Save or load a nested structure."""

import logging
import os.path
import re

import tensorflow as tf


def latest_checkpoint(root_output_dir, checkpoint_prefix='ckpt_'):
  r"""Get the latest checkpoint name.

  Searches `root_output_dir` for directories matching the regular expression
  `checkpoint_prefix_\d+$` and returns the directory with the largest integer
  suffix.

  Args:
    root_output_dir: The directory where all checkpoints stored.
    checkpoint_prefix: The common prefix shared by all checkpoint directories.

  Returns:
    Dirname of the lastest checkpoint.
  """
  checkpoints = tf.io.gfile.glob(
      os.path.join(root_output_dir, '{}*'.format(checkpoint_prefix)))
  if not checkpoints:
    return None

  checkpoint_regex = re.compile(
      r'^(?P<prefix>{})(?P<num>\d+)$'.format(checkpoint_prefix))

  max_checkpoint_path = None
  max_checkpoint_num = -1
  for checkpoint_path in checkpoints:
    matcher = checkpoint_regex.match(os.path.basename(checkpoint_path))
    if not matcher:
      continue
    checkpoint_num = int(matcher.group('num'))
    if checkpoint_num > max_checkpoint_num:
      max_checkpoint_path = checkpoint_path
      max_checkpoint_num = checkpoint_num
  return max_checkpoint_path


def save(obj, export_dir):
  r"""Save a nested structure to `export_dir`.

  NOTE: to be compatible with `latest_checkpoint`, the basename of `export_dir`
  must follow the regular expression pattern `<prefix>\d+`, where the final
  digit matcher  determines the ordering of the checkpoints.

  Args:
    obj: A nested structure which `tf.convert_to_tensor` supports.
    export_dir: A directory in which to write the state.
  """
  model = tf.Module()
  model.obj = tf.nest.flatten(obj)
  model.build_obj_fn = tf.function(lambda: model.obj, input_signature=())
  tf.saved_model.save(model, export_dir, signatures={})
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
