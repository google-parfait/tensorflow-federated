# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for intrinsic_bodies.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_bodies


def _body_str(comp):
  """Returns the string representation of `comp`'s body."""
  return str(
      computation_building_blocks.ComputationBuildingBlock.from_proto(
          computation_impl.ComputationImpl.get_proto(comp)))


class IntrinsicBodiesTest(absltest.TestCase):

  def test_federated_sum(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      return bodies['federated_sum'](x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32@SERVER)')
    self.assertEqual(
        _body_str(foo),
        '(FEDERATED_arg -> federated_reduce(<FEDERATED_arg,'
        'generic_zero,generic_plus>))')


if __name__ == '__main__':
  absltest.main()
