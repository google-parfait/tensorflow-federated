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
"""Tests for ComputationImpl."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import type_serialization


class ComputationImplTest(absltest.TestCase):

  def test_something(self):
    # TODO(b/113112108): Revise these tests after a more complete implementation
    # is in place.

    # At the moment, this should succeed, as both the computation body and the
    # type are well-formed.
    computation_impl.ComputationImpl(
        pb.Computation(
            **{
                'type':
                    type_serialization.serialize_type(
                        computation_types.FunctionType(tf.int32, tf.int32)),
                'intrinsic':
                    pb.Intrinsic(uri='whatever')
            }), context_stack_impl.context_stack)

    # This should fail, as the proto is not well-formed.
    self.assertRaises(TypeError, computation_impl.ComputationImpl,
                      pb.Computation(), context_stack_impl.context_stack)

    # This should fail, as "10" is not an instance of pb.Computation.
    self.assertRaises(TypeError, computation_impl.ComputationImpl, 10,
                      context_stack_impl.context_stack)


if __name__ == '__main__':
  absltest.main()
