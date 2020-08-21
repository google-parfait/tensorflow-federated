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

import asyncio

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_factory


def create_test_executor_factory():
  executor = eager_tf_executor.EagerTFExecutor()
  executor = reference_resolving_executor.ReferenceResolvingExecutor(executor)
  return executor_factory.ExecutorFactoryImpl(lambda _: executor)


class ReferenceResolvingExecutorTest(absltest.TestCase):

  def test_with_no_arg_tf_comp_in_no_arg_fed_comp(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    @computations.federated_computation
    def comp():
      return 10

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(ex.create_call(v1))
    result = loop.run_until_complete(v2.compute())
    self.assertEqual(result.numpy(), 10)

  def test_with_one_arg_tf_comp_in_no_arg_fed_comp(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation
    def comp():
      return add_one(10)

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(ex.create_call(v1))
    result = loop.run_until_complete(v2.compute())
    self.assertEqual(result.numpy(), 11)

  def test_clear_failure_with_mismatched_types_in_create_call(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    @computations.federated_computation(tf.float32)
    def comp(x):
      return x

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(ex.create_value(10, tf.int32))
    with self.assertRaisesRegex(TypeError, 'incompatible'):
      loop.run_until_complete(ex.create_call(v1, v2))

  def test_with_one_arg_tf_comp_in_one_arg_fed_comp(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation(tf.int32)
    def comp(x):
      return add_one(add_one(x))

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(ex.create_value(10, tf.int32))
    v3 = loop.run_until_complete(ex.create_call(v1, v2))
    result = loop.run_until_complete(v3.compute())
    self.assertEqual(result.numpy(), 12)

  def test_with_one_arg_tf_comp_in_two_arg_fed_comp(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32, tf.int32)
    def add_numbers(x, y):
      return x + y

    @computations.federated_computation(tf.int32, tf.int32)
    def comp(x, y):
      return add_numbers(x, x), add_numbers(x, y), add_numbers(y, y)

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(ex.create_value(10, tf.int32))
    v3 = loop.run_until_complete(ex.create_value(20, tf.int32))
    v4 = loop.run_until_complete(
        ex.create_struct(structure.Struct([('x', v2), ('y', v3)])))
    v5 = loop.run_until_complete(ex.create_call(v1, v4))
    result = loop.run_until_complete(v5.compute())
    self.assertEqual(
        str(structure.map_structure(lambda x: x.numpy(), result)), '<20,30,40>')

  def test_with_functional_parameter(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation(
        computation_types.FunctionType(tf.int32, tf.int32), tf.int32)
    def comp(f, x):
      return f(f(x))

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(ex.create_value(add_one))
    v3 = loop.run_until_complete(ex.create_value(10, tf.int32))
    v4 = loop.run_until_complete(
        ex.create_struct(structure.Struct([('f', v2), ('x', v3)])))
    v5 = loop.run_until_complete(ex.create_call(v1, v4))
    result = loop.run_until_complete(v5.compute())
    self.assertEqual(result.numpy(), 12)

  def test_with_tuples(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32, tf.int32)
    def add_numbers(x, y):
      return x + y

    @computations.federated_computation
    def comp():
      return add_numbers(10, 20)

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(ex.create_call(v1))
    result = loop.run_until_complete(v2.compute())
    self.assertEqual(result.numpy(), 30)

  def test_create_selection_with_tuples(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    v1 = loop.run_until_complete(ex.create_value(10, tf.int32))
    v2 = loop.run_until_complete(ex.create_value(20, tf.int32))
    v3 = loop.run_until_complete(
        ex.create_struct(structure.Struct([(None, v1), (None, v2)])))
    v4 = loop.run_until_complete(ex.create_selection(v3, index=0))
    v5 = loop.run_until_complete(ex.create_selection(v3, index=1))
    result0 = loop.run_until_complete(v4.compute())
    result1 = loop.run_until_complete(v5.compute())
    self.assertEqual(result0.numpy(), 10)
    self.assertEqual(result1.numpy(), 20)

  def test_with_nested_lambdas(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32, tf.int32)
    def add_numbers(x, y):
      return x + y

    @computations.federated_computation(tf.int32)
    def comp(x):

      @computations.federated_computation(tf.int32)
      def nested_comp(y):
        return add_numbers(x, y)

      return nested_comp(1)

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(ex.create_value(10, tf.int32))
    v3 = loop.run_until_complete(ex.create_call(v1, v2))
    result = loop.run_until_complete(v3.compute())
    self.assertEqual(result.numpy(), 11)

  def test_with_block(self):
    ex = reference_resolving_executor.ReferenceResolvingExecutor(
        eager_tf_executor.EagerTFExecutor())
    loop = asyncio.get_event_loop()

    f_type = computation_types.FunctionType(tf.int32, tf.int32)
    a = building_blocks.Reference(
        'a', computation_types.StructType([('f', f_type), ('x', tf.int32)]))
    ret = building_blocks.Block([('f', building_blocks.Selection(a, name='f')),
                                 ('x', building_blocks.Selection(a, name='x'))],
                                building_blocks.Call(
                                    building_blocks.Reference('f', f_type),
                                    building_blocks.Call(
                                        building_blocks.Reference('f', f_type),
                                        building_blocks.Reference(
                                            'x', tf.int32))))
    comp = building_blocks.Lambda(a.name, a.type_signature, ret)

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    v1 = loop.run_until_complete(
        ex.create_value(comp.proto, comp.type_signature))
    v2 = loop.run_until_complete(ex.create_value(add_one))
    v3 = loop.run_until_complete(ex.create_value(10, tf.int32))
    v4 = loop.run_until_complete(
        ex.create_struct(structure.Struct([('f', v2), ('x', v3)])))
    v5 = loop.run_until_complete(ex.create_call(v1, v4))
    result = loop.run_until_complete(v5.compute())
    self.assertEqual(result.numpy(), 12)

  def test_with_federated_map(self):
    eager_ex = eager_tf_executor.EagerTFExecutor()
    factory = federated_resolving_strategy.FederatedResolvingStrategy.factory(
        {placement_literals.SERVER: eager_ex})
    federated_ex = federating_executor.FederatingExecutor(factory, eager_ex)
    ex = reference_resolving_executor.ReferenceResolvingExecutor(federated_ex)
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation(type_factory.at_server(tf.int32))
    def comp(x):
      return intrinsics.federated_map(add_one, x)

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(
        ex.create_value(10, type_factory.at_server(tf.int32)))
    v3 = loop.run_until_complete(ex.create_call(v1, v2))
    result = loop.run_until_complete(v3.compute())
    self.assertEqual(result.numpy(), 11)

  def test_with_federated_map_and_broadcast(self):
    eager_ex = eager_tf_executor.EagerTFExecutor()
    factory = federated_resolving_strategy.FederatedResolvingStrategy.factory({
        placement_literals.SERVER: eager_ex,
        placement_literals.CLIENTS: [eager_ex for _ in range(3)]
    })
    federated_ex = federating_executor.FederatingExecutor(factory, eager_ex)
    ex = reference_resolving_executor.ReferenceResolvingExecutor(federated_ex)
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation(type_factory.at_server(tf.int32))
    def comp(x):
      return intrinsics.federated_map(add_one,
                                      intrinsics.federated_broadcast(x))

    v1 = loop.run_until_complete(ex.create_value(comp))
    v2 = loop.run_until_complete(
        ex.create_value(10, type_factory.at_server(tf.int32)))
    v3 = loop.run_until_complete(ex.create_call(v1, v2))
    result = loop.run_until_complete(v3.compute())
    self.assertCountEqual([x.numpy() for x in result], [11, 11, 11])

  def test_raises_with_closure(self):
    eager_ex = eager_tf_executor.EagerTFExecutor()
    factory = federated_resolving_strategy.FederatedResolvingStrategy.factory({
        placement_literals.SERVER: eager_ex,
    })
    federated_ex = federating_executor.FederatingExecutor(factory, eager_ex)
    ex = reference_resolving_executor.ReferenceResolvingExecutor(federated_ex)
    loop = asyncio.get_event_loop()

    @computations.federated_computation(tf.int32,
                                        type_factory.at_server(tf.int32))
    def foo(x, y):

      @computations.federated_computation(tf.int32)
      def bar(z):
        del z
        return x

      return intrinsics.federated_map(bar, y)

    v1 = loop.run_until_complete(ex.create_value(foo))
    v2 = loop.run_until_complete(
        ex.create_value(
            structure.Struct([('x', 0), ('y', 0)]),
            [tf.int32, type_factory.at_server(tf.int32)]))
    with self.assertRaisesRegex(
        RuntimeError,
        'lambda passed to intrinsic contains references to captured variables'):
      loop.run_until_complete(ex.create_call(v1, v2))

  def test_execution_of_tensorflow(self):

    @computations.tf_computation
    def comp():
      return tf.math.add(5, 5)

    executor = create_test_executor_factory()
    with executor_test_utils.install_executor(executor):
      result = comp()

    self.assertEqual(result, 10)


if __name__ == '__main__':
  absltest.main()
