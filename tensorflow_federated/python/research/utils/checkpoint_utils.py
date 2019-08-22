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
import os

import numpy as np
import tensorflow as tf


def latest_checkpoint(root_output_dir):
  """Get the latest checkpoint dirname, which is in the format of `ckpt_1`.

  Args:
    root_output_dir: The directory where all checkpoints stored.

  Returns:
    Dirname of the lastest checkpoint.
  """
  checkpoints = tf.io.gfile.glob(os.path.join(root_output_dir, 'ckpt_*'))
  if not checkpoints:
    return None

  lastest_index = np.argmax([
      int(dirname.split('_')[-1])
      for dirname in checkpoints
      if tf.io.gfile.isdir(dirname)
  ])

  return checkpoints[lastest_index]


def save(obj, export_dir):
  """Save a nested structure to export_dir.

  Args:
    obj: A nested structure which `tf.convert_to_tensor` supports.
    export_dir: A directory in which to write the state.
  """
  model = tf.Module()
  model.obj = tf.nest.flatten(obj)
  model.build_obj_fn = tf.function(lambda: model.obj, input_signature=())
  tf.saved_model.save(model, export_dir, signatures={})

  logging.info('A nested structure is saved to %s.', export_dir)


def load(export_dir, obj_template):
  """Load a nested structure from export_dir.

  Args:
    export_dir: The directory to load from.
    obj_template: An object that provides the nested structure to mimic.

  Returns:
    Loaded nested structure.

  Raises:
    FileNotFoundError: No such file or directory.
  """

  if tf.io.gfile.exists(export_dir):
    loaded = tf.saved_model.load_v2(export_dir)

    flat_obj = loaded.build_obj_fn()
    obj = tf.nest.pack_sequence_as(obj_template, flat_obj)

    tf.logging.info('Load a nested structure from %s!' % export_dir)
  else:
    raise FileNotFoundError('No such file or directory: %s.' % export_dir)

  return obj
