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
"""Tests for proto_transformations.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.impl import computation_building_block_utils
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_wrapper_instances
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import proto_transformations
from tensorflow_federated.python.core.impl import tensorflow_serialization


def _create_compiled_computation(py_fn, arg_type):
  proto, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      py_fn, arg_type, context_stack_impl.context_stack)
  return computation_building_blocks.CompiledComputation(proto)


class PruneTensorFlowProtoTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      proto_transformations.prune_tensorflow_proto(None)

  def test_raises_on_compiled_computation(self):

    def fn(x):
      return x

    comp = _create_compiled_computation(fn, tf.int32)
    with self.assertRaises(TypeError):
      proto_transformations.prune_tensorflow_proto(comp)

  def test_does_not_reduce_no_unnecessary_ops(self):

    def fn(x):
      return x

    comp = _create_compiled_computation(fn, tf.int32)
    pruned = computation_building_blocks.CompiledComputation(
        proto_transformations.prune_tensorflow_proto(comp.proto))
    ops_before = computation_building_block_utils.count_tensorflow_ops_in(comp)
    ops_after = computation_building_block_utils.count_tensorflow_ops_in(pruned)
    self.assertEqual(ops_before, ops_after)

  def test_reduces_unnecessary_ops(self):

    def bad_fn(x):
      _ = tf.constant(0)
      return x

    comp = _create_compiled_computation(bad_fn, tf.int32)
    ops_before = computation_building_block_utils.count_tensorflow_ops_in(comp)
    reduced_proto = proto_transformations.prune_tensorflow_proto(comp.proto)
    reduced_comp = computation_building_blocks.CompiledComputation(
        reduced_proto)
    ops_after = computation_building_block_utils.count_tensorflow_ops_in(
        reduced_comp)
    self.assertLess(ops_after, ops_before)

  def test_prune_does_not_change_exeuction(self):

    def bad_fn(x):
      _ = tf.constant(0)
      return x

    comp = _create_compiled_computation(bad_fn, tf.int32)
    reduced_proto = proto_transformations.prune_tensorflow_proto(comp.proto)
    reduced_comp = computation_building_blocks.CompiledComputation(
        reduced_proto)

    orig_executable = computation_wrapper_instances.building_block_to_computation(
        comp)
    reduced_executable = computation_wrapper_instances.building_block_to_computation(
        reduced_comp)
    for k in range(5):
      self.assertEqual(orig_executable(k), reduced_executable(k))


if __name__ == '__main__':
  absltest.main()
