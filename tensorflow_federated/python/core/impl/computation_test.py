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

import unittest

from tensorflow_federated.proto.v0 import computation_pb2 as pb

from tensorflow_federated.python.core.impl import computation_impl as ci


class ComputationTest(unittest.TestCase):

  def test_something(self):
    # A hypothetical example of a federated computation definition in Python,
    # expressed in a yet-to-be-defined syntax.
    #
    # @tff.computation
    # def fed_eval(model):
    #
    #   @tfe.defun
    #   def local_eval(model):
    #     ...
    #     return {'loss': ..., 'accuracy': ...}
    #
    #   client_model = tff.federated_broadcast(model)
    #   client_metrics = tff.federated_map(local_eval, client_model)
    #   return tff.federated_average(client_metrics)
    #
    # The corresponding representation as computation.proto.
    fed_eval = pb.Computation(**{'lambda': pb.Lambda(
        parameter_name='model',
        result=pb.Computation(block=pb.Block(
            local=[
                pb.Block.Local(name='local_eval', value=pb.Computation(
                    tensorflow=pb.TensorFlow())),
                pb.Block.Local(name='client_model', value=pb.Computation(
                    call=pb.Call(
                        function=pb.Computation(
                            intrinsic=pb.Intrinsic(uri='federated_broadcast')),
                        argument=pb.Computation(
                            reference=pb.Reference(name='model'))))),
                pb.Block.Local(name='client_metrics', value=pb.Computation(
                    call=pb.Call(
                        function=pb.Computation(
                            intrinsic=pb.Intrinsic(uri='federated_map')),
                        argument=pb.Computation(
                            tuple=pb.Tuple(element=[
                                pb.Tuple.Element(
                                    value=pb.Computation(
                                        reference=pb.Reference(
                                            name='local_eval'))),
                                pb.Tuple.Element(
                                    value=pb.Computation(
                                        reference=pb.Reference(
                                            name='local_client_model')))])))))],
            result=pb.Computation(
                call=pb.Call(
                    function=pb.Computation(
                        intrinsic=pb.Intrinsic(uri='federated_average')),
                    argument=pb.Computation(
                        reference=pb.Reference(name='client_metrics')))))))})
    ci.ComputationImpl(fed_eval)

    # This will successfully construct a lambda "x -> x.func(x.arg)".
    ci.ComputationImpl(pb.Computation(**{'lambda': pb.Lambda(
        parameter_name='x', result=pb.Computation(call=pb.Call(
            function=pb.Computation(selection=pb.Selection(
                source=pb.Computation(reference=pb.Reference(name='x')),
                name='func')),
            argument=pb.Computation(selection=pb.Selection(
                source=pb.Computation(reference=pb.Reference(name='x')),
                name='arg')))))}))

    ci.ComputationImpl(pb.Computation(intrinsic=pb.Intrinsic(uri='broadcast')))

    # This should fail, as "10" is not an instance of pb.Computation.
    self.assertRaises(TypeError, ci.ComputationImpl, 10)


if __name__ == '__main__':
  unittest.main()
