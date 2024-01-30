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

from absl.testing import absltest
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import polymorphic_computation
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization


class PolymorphicComputationTest(absltest.TestCase):

  def test_call_returns_result(self):
    class TestContext(context_base.SyncContext):

      def ingest(self, val, type_spec):
        del type_spec  # Unused.
        return val

      def invoke(self, comp, arg):
        return 'name={},type={},arg={},unpack={}'.format(
            comp.name, comp.type_signature.parameter, arg, comp.unpack
        )

    class TestContextStack(context_stack_base.ContextStack):

      def __init__(self):
        super().__init__()
        self._context = TestContext()

      @property
      def current(self):
        return self._context

      def install(self, ctx):
        del ctx  # Unused
        return self._context

    context_stack = TestContextStack()

    class TestFunction(computation_impl.ConcreteComputation):

      def __init__(self, name, unpack, parameter_type):
        self._name = name
        self._unpack = unpack
        type_signature = computation_types.FunctionType(parameter_type, np.str_)
        test_proto = pb.Computation(
            type=type_serialization.serialize_type(type_signature)
        )
        super().__init__(test_proto, context_stack, type_signature)

      @property
      def name(self):
        return self._name

      @property
      def unpack(self):
        return self._unpack

    class TestFunctionFactory:

      def __init__(self):
        self._count = 0

      def __call__(self, parameter_type, unpack):
        self._count = self._count + 1
        return TestFunction(str(self._count), str(unpack), parameter_type)

    fn = polymorphic_computation.PolymorphicComputation(
        TestFunctionFactory(), type_conversions.infer_type
    )

    self.assertEqual(fn(10), 'name=1,type=<int32>,arg=<10>,unpack=True')
    self.assertEqual(
        fn(20, x=True), 'name=2,type=<int32,x=bool>,arg=<20,x=True>,unpack=True'
    )
    fn_with_bool_arg = fn.fn_for_argument_type(
        computation_types.to_type(np.bool_)
    )
    self.assertEqual(
        fn_with_bool_arg(True), 'name=3,type=bool,arg=True,unpack=None'
    )
    self.assertEqual(
        fn(30, x=40), 'name=4,type=<int32,x=int32>,arg=<30,x=40>,unpack=True'
    )
    self.assertEqual(fn(50), 'name=1,type=<int32>,arg=<50>,unpack=True')
    self.assertEqual(
        fn(0, x=False), 'name=2,type=<int32,x=bool>,arg=<0,x=False>,unpack=True'
    )
    fn_with_bool_arg = fn.fn_for_argument_type(
        computation_types.to_type(np.bool_)
    )
    self.assertEqual(
        fn_with_bool_arg(False), 'name=3,type=bool,arg=False,unpack=None'
    )
    self.assertEqual(
        fn(60, x=70), 'name=4,type=<int32,x=int32>,arg=<60,x=70>,unpack=True'
    )


if __name__ == '__main__':
  absltest.main()
