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
"""Tests for ServerState save."""

import functools
import os

import attr
import tensorflow as tf

import tensorflow_federated as tff
from tensorflow_federated.python.examples.mnist import models
from tensorflow_federated.python.research.utils import checkpoint_utils


@attr.s(cmp=False, frozen=False)
class Obj(object):
  """Container for all state that need to be stored in the checkpoint.

  Attributes:
    model: A ModelWeights structure, containing Tensors or Variables.
    optimizer_state: A list of Tensors or Variables, in the order returned by
      optimizer.variables().
    round_num: Training round_num.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()

  @classmethod
  def from_anon_tuple(cls, anon_tuple, round_num):
    # TODO(b/130724878): These conversions should not be needed.
    return cls(
        model=anon_tuple.model._asdict(recursive=True),
        optimizer_state=list(anon_tuple.optimizer_state),
        round_num=round_num)


class SavedStateTest(tf.test.TestCase):

  def test_save_and_load(self):
    server_optimizer_fn = functools.partial(
        tf.keras.optimizers.SGD, learning_rate=0.1, momentum=0.9)

    iterative_process = tff.learning.build_federated_averaging_process(
        models.model_fn, server_optimizer_fn=server_optimizer_fn)
    server_state = iterative_process.initialize()
    # TODO(b/130724878): These conversions should not be needed.
    obj = Obj.from_anon_tuple(server_state, 1)

    export_dir = os.path.join(self.get_temp_dir(), 'ckpt_1')
    checkpoint_utils.save(obj, export_dir)

    loaded_obj = checkpoint_utils.load(export_dir, obj)

    self.assertAllClose(tf.nest.flatten(obj), tf.nest.flatten(loaded_obj))

  def test_load_latest_state(self):
    server_optimizer_fn = functools.partial(
        tf.keras.optimizers.SGD, learning_rate=0.1, momentum=0.9)

    iterative_process = tff.learning.build_federated_averaging_process(
        models.model_fn, server_optimizer_fn=server_optimizer_fn)
    server_state = iterative_process.initialize()
    # TODO(b/130724878): These conversions should not be needed.
    obj_1 = Obj.from_anon_tuple(server_state, 1)
    export_dir = os.path.join(self.get_temp_dir(), 'ckpt_1')
    checkpoint_utils.save(obj_1, export_dir)

    # TODO(b/130724878): These conversions should not be needed.
    obj_2 = Obj.from_anon_tuple(server_state, 2)
    export_dir = os.path.join(self.get_temp_dir(), 'ckpt_2')
    checkpoint_utils.save(obj_2, export_dir)

    export_dir = checkpoint_utils.latest_checkpoint(self.get_temp_dir())

    loaded_obj = checkpoint_utils.load(export_dir, obj_1)

    self.assertEqual(os.path.join(self.get_temp_dir(), 'ckpt_2'), export_dir)
    self.assertAllClose(tf.nest.flatten(obj_2), tf.nest.flatten(loaded_obj))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
