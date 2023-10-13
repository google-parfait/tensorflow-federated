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

from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.context_stack import get_context_stack
from tensorflow_federated.python.core.impl.context_stack import runtime_error_context
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.types import computation_types


class FederatedComputationWrapperTest(absltest.TestCase):

  def test_federated_computation_wrapper(self):

    @federated_computation.federated_computation(
        (computation_types.FunctionType(np.int32, np.int32), np.int32)
    )
    def foo(f, x):
      return f(f(x))

    self.assertIsInstance(foo, computation_impl.ConcreteComputation)
    self.assertEqual(
        str(foo.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)'
    )

    self.assertEqual(
        str(foo.to_building_block()),
        (
            '(foo_arg -> (let'
            ' fc_foo_symbol_0=foo_arg[0](foo_arg[1]),fc_foo_symbol_1=foo_arg[0](fc_foo_symbol_0)'
            ' in fc_foo_symbol_1))'
        ),
    )

  def test_stackframes_in_errors(self):
    class DummyError(RuntimeError):
      pass

    with self.assertRaises(DummyError):
      @federated_computation.federated_computation
      def _():
        raise DummyError()

  def test_empty_tuple_arg(self):

    @federated_computation.federated_computation(
        computation_types.StructType([])
    )
    def foo(x):
      return x

    self.assertIsInstance(foo, computation_impl.ConcreteComputation)
    self.assertEqual(str(foo.type_signature), '(<> -> <>)')

    self.assertEqual(str(foo.to_building_block()), '(foo_arg -> foo_arg)')

  def test_stack_resets_on_none_returned(self):
    stack = get_context_stack.get_context_stack()
    self.assertIsInstance(
        stack.current, runtime_error_context.RuntimeErrorContext
    )

    with self.assertRaises(computation_wrapper.ComputationReturnedNoneError):
      @federated_computation.federated_computation()
      def _():
        pass

    self.assertIsInstance(
        stack.current, runtime_error_context.RuntimeErrorContext
    )


if __name__ == '__main__':
  absltest.main()
