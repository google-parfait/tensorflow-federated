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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import test_utils as common_libs_test_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import intrinsic_bodies
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.types import placement_literals


@executor_test_utils.executors
class IntrinsicBodiesTest(common_libs_test_utils.TestCase,
                          parameterized.TestCase):

  def test_federated_sum(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.FEDERATED_SUM.uri](x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32@SERVER)')

    self.assertEqual(foo([1]), 1)
    self.assertEqual(foo([1, 2, 3]), 6)

  def test_federated_sum_named_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([('a', tf.int32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.FEDERATED_SUM.uri](x)

    self.assertEqual(
        str(foo.type_signature),
        '({<a=int32,b=float32>}@CLIENTS -> <a=int32,b=float32>@SERVER)')
    self.assertDictEqual(structure.to_odict(foo([[1, 2.]])), {'a': 1, 'b': 2.})
    self.assertDictEqual(
        structure.to_odict(foo([[1, 2.], [3, 4.]])), {
            'a': 4,
            'b': 6.
        })

  def test_federated_weighted_mean_with_ints(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> float64@SERVER)')

    self.assertEqual(foo([1]), 1.)
    self.assertEqual(foo([1, 2, 3]), 14. / 6)

  def test_federated_weighted_mean_named_tuple_with_tensor(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([[('a', tf.float32),
                                          ('b', tf.float32)], tf.float32],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri](x)

    self.assertEqual(
        str(foo.type_signature),
        '({<<a=float32,b=float32>,float32>}@CLIENTS -> <a=float32,b=float32>@SERVER)'
    )

    self.assertEqual(
        foo([[[1., 1.], 1.]]), structure.Struct([('a', 1.), ('b', 1.)]))
    self.assertEqual(
        foo([[[1., 1.], 1.], [[1., 2.], 2.], [[1., 4.], 4.]]),
        structure.Struct([('a', 1.), ('b', 3.)]))

  def test_federated_mean_with_ints(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.FEDERATED_MEAN.uri](x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> float64@SERVER)')

    self.assertEqual(foo([1]), 1.)
    self.assertEqual(foo([1, 2, 3]), 2.)

  def test_federated_mean_named_tuple_with_tensor(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([('a', tf.float32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.FEDERATED_MEAN.uri](x)

    self.assertEqual(
        str(foo.type_signature),
        '({<a=float32,b=float32>}@CLIENTS -> <a=float32,b=float32>@SERVER)')

    self.assertEqual(foo([[1., 1.]]), structure.Struct([('a', 1.), ('b', 1.)]))
    self.assertEqual(
        foo([[1., 1.], [1., 2.], [1., 3.]]),
        structure.Struct([('a', 1.), ('b', 2.)]))


@executor_test_utils.executors
class GenericDivideTest(common_libs_test_utils.TestCase,
                        parameterized.TestCase):

  def test_generic_divide_unplaced_named_tuple_by_tensor(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([[('a', tf.float32),
                                          ('b', tf.float32)], tf.float32],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_DIVIDE.uri]([x[0], x[1]])

    self.assertEqual(
        str(foo.type_signature),
        '({<<a=float32,b=float32>,float32>}@CLIENTS -> {<a=float32,b=float32>}@CLIENTS)'
    )

    self.assertEqual(
        foo([[[1., 1.], 1.]]), [structure.Struct([('a', 1.), ('b', 1.)])])
    self.assertEqual(
        foo([[[1., 1.], 1.], [[1., 2.], 2.], [[1., 4.], 4.]]), [
            structure.Struct([('a', 1.), ('b', 1.)]),
            structure.Struct([('a', 0.5), ('b', 1.)]),
            structure.Struct([('a', 0.25), ('b', 1.)])
        ])

  def test_generic_divide_with_unplaced_scalars(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(tf.float32)
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_DIVIDE.uri]([x, x])

    self.assertEqual(str(foo.type_signature), '(float32 -> float32)')

    self.assertEqual(foo(1.), 1.)
    self.assertEqual(foo(2.), 1.)
    self.assertEqual(foo(3.), 1.)

  def test_generic_divide_with_unplaced_named_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_DIVIDE.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '(<a=int32,b=float32> -> <a=float64,b=float32>)')

    self.assertEqual(foo([1, 1.]), structure.Struct([('a', 1.), ('b', 1.)]))

  def test_generic_divide_with_unplaced_named_tuple_and_tensor(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.StructType([[('a', tf.float32), ('b', tf.float32)],
                                      tf.float32]))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_DIVIDE.uri](x)

    self.assertEqual(
        str(foo.type_signature),
        '(<<a=float32,b=float32>,float32> -> <a=float32,b=float32>)')

    self.assertEqual(
        foo([[1., 1.], 2.]), structure.Struct([('a', .5), ('b', .5)]))

  def test_generic_divide_with_named_tuple_of_federated_types(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    fed_int = computation_types.FederatedType(tf.int32,
                                              placement_literals.CLIENTS)

    @computations.federated_computation([('a', fed_int), ('b', fed_int)])
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_DIVIDE.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '(<a={int32}@CLIENTS,b={int32}@CLIENTS> -> <a={float64}@CLIENTS,b={float64}@CLIENTS>)'
    )

    self.assertEqual(
        foo([[1], [1]]), structure.Struct([('a', [1.]), ('b', [1.])]))

  def test_federated_generic_divide_with_federated_named_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([('a', tf.int32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_DIVIDE.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '({<a=int32,b=float32>}@CLIENTS -> {<a=float64,b=float32>}@CLIENTS)')

    self.assertEqual(foo([[1, 1.]]), [structure.Struct([('a', 1.), ('b', 1.)])])
    self.assertEqual(
        foo([[1, 1.], [1, 2.], [3, 3.]]),
        [structure.Struct([('a', 1.), ('b', 1.)])] * 3)

  def test_federated_generic_divide_with_ints(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_DIVIDE.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> {float64}@CLIENTS)')

    self.assertEqual(foo([1]), [1.])
    self.assertEqual(foo([1, 2, 3]), [1., 1., 1.])

  def test_federated_generic_divide_with_unnamed_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([tf.int32, tf.float32],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_DIVIDE.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '({<int32,float32>}@CLIENTS -> {<float64,float32>}@CLIENTS)')

    self.assertEqual(
        foo([[1, 1.]]), [structure.Struct([(None, 1.), (None, 1.)])])
    self.assertEqual(
        foo([[1, 1.], [1, 2.], [3, 3.]]),
        [structure.Struct([(None, 1.), (None, 1.)])] * 3)


@executor_test_utils.executors
class GenericMultiplyTest(common_libs_test_utils.TestCase,
                          parameterized.TestCase):

  def test_generic_multiply_federated_named_tuple_by_tensor(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([[('a', tf.float32),
                                          ('b', tf.float32)], tf.float32],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_MULTIPLY.uri]([x[0], x[1]])

    self.assertEqual(
        str(foo.type_signature),
        '({<<a=float32,b=float32>,float32>}@CLIENTS -> {<a=float32,b=float32>}@CLIENTS)'
    )

    self.assertEqual(
        foo([[[1., 1.], 1.]]), [structure.Struct([('a', 1.), ('b', 1.)])])
    self.assertEqual(
        foo([[[1., 1.], 1.], [[1., 2.], 2.], [[1., 4.], 4.]]), [
            structure.Struct([('a', 1.), ('b', 1.)]),
            structure.Struct([('a', 2.), ('b', 4.)]),
            structure.Struct([('a', 4.), ('b', 16.)])
        ])

  def test_generic_multiply_with_unplaced_scalars(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(tf.float32)
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_MULTIPLY.uri]([x, x])

    self.assertEqual(str(foo.type_signature), '(float32 -> float32)')

    self.assertEqual(foo(1.), 1.)
    self.assertEqual(foo(2.), 4.)
    self.assertEqual(foo(3.), 9.)

  def test_federated_generic_multiply_with_ints(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_MULTIPLY.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> {int32}@CLIENTS)')

    self.assertEqual(foo([1]), [1])
    self.assertEqual(foo([1, 2, 3]), [1, 4, 9])

  def test_generic_multiply_with_unplaced_named_tuple_and_tensor(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.StructType([[('a', tf.float32), ('b', tf.float32)],
                                      tf.float32]))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_MULTIPLY.uri](x)

    self.assertEqual(
        str(foo.type_signature),
        '(<<a=float32,b=float32>,float32> -> <a=float32,b=float32>)')

    self.assertEqual(
        foo([[1., 1.], 2.]), structure.Struct([('a', 2.), ('b', 2.)]))

  def test_federated_generic_multiply_with_unnamed_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([tf.int32, tf.float32],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_MULTIPLY.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '({<int32,float32>}@CLIENTS -> {<int32,float32>}@CLIENTS)')

    self.assertEqual(
        foo([[1, 1.]]), [structure.Struct([(None, 1), (None, 1.)])])
    self.assertEqual(
        foo([[1, 1.], [1, 2.], [1, 3.]]), [
            structure.Struct([(None, 1), (None, 1.)]),
            structure.Struct([(None, 1), (None, 4.)]),
            structure.Struct([(None, 1), (None, 9.)])
        ])

  def test_federated_generic_multiply_with_named_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([('a', tf.int32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_MULTIPLY.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '({<a=int32,b=float32>}@CLIENTS -> {<a=int32,b=float32>}@CLIENTS)')

    self.assertEqual(foo([[1, 1.]]), [structure.Struct([('a', 1), ('b', 1.)])])
    self.assertEqual(
        foo([[1, 1.], [1, 2.], [1, 3.]]), [
            structure.Struct([('a', 1), ('b', 1.)]),
            structure.Struct([('a', 1), ('b', 4.)]),
            structure.Struct([('a', 1), ('b', 9.)])
        ])

  def test_generic_multiply_with_named_tuple_of_federated_types(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    fed_int = computation_types.FederatedType(tf.int32,
                                              placement_literals.CLIENTS)

    @computations.federated_computation([('a', fed_int), ('b', fed_int)])
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_MULTIPLY.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '(<a={int32}@CLIENTS,b={int32}@CLIENTS> -> <a={int32}@CLIENTS,b={int32}@CLIENTS>)'
    )

    self.assertEqual(
        foo([[1], [1]]), structure.Struct([('a', [1]), ('b', [1])]))

  def test_generic_multiply_with_unplaced_named_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.StructType([('a', tf.float32), ('b', tf.float32)]))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_MULTIPLY.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '(<a=float32,b=float32> -> <a=float32,b=float32>)')

    self.assertEqual(foo([1., 1.]), structure.Struct([('a', 1.), ('b', 1.)]))


@executor_test_utils.executors
class GenericAddTest(common_libs_test_utils.TestCase, parameterized.TestCase):

  def test_federated_generic_add_with_ints(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> {int32}@CLIENTS)')

    self.assertEqual(foo([1]), [2])
    self.assertEqual(foo([1, 2, 3]), [2, 4, 6])

  def test_federated_generic_add_with_unnamed_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([tf.int32, tf.float32],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '({<int32,float32>}@CLIENTS -> {<int32,float32>}@CLIENTS)')

    self.assertEqual(
        foo([[1, 1.]]), [structure.Struct([(None, 2), (None, 2.)])])
    self.assertEqual(
        foo([[1, 1.], [1, 2.], [1, 3.]]), [
            structure.Struct([(None, 2), (None, 2.)]),
            structure.Struct([(None, 2), (None, 4.)]),
            structure.Struct([(None, 2), (None, 6.)])
        ])

  def test_federated_generic_add_with_named_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([('a', tf.int32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '({<a=int32,b=float32>}@CLIENTS -> {<a=int32,b=float32>}@CLIENTS)')

    self.assertEqual(foo([[1, 1.]]), [structure.Struct([('a', 2), ('b', 2.)])])
    self.assertEqual(
        foo([[1, 1.], [1, 2.], [1, 3.]]), [
            structure.Struct([('a', 2), ('b', 2.)]),
            structure.Struct([('a', 2), ('b', 4.)]),
            structure.Struct([('a', 2), ('b', 6.)])
        ])

  def test_generic_add_with_named_tuple_of_federated_types(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    fed_int = computation_types.FederatedType(tf.int32,
                                              placement_literals.CLIENTS)

    @computations.federated_computation([('a', fed_int), ('b', fed_int)])
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature),
        '(<a={int32}@CLIENTS,b={int32}@CLIENTS> -> <a={int32}@CLIENTS,b={int32}@CLIENTS>)'
    )

    self.assertEqual(
        foo([[1], [1]]),
        structure.Struct([('a', tf.constant([2.])), ('b', tf.constant([2.]))]))

  def test_generic_add_with_unplaced_named_tuples(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.StructType([('a', tf.int32), ('b', tf.float32)]))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x, x])

    self.assertEqual(
        str(foo.type_signature), '(<a=int32,b=float32> -> <a=int32,b=float32>)')

    self.assertEqual(foo([1, 1.]), structure.Struct([('a', 2), ('b', 2.)]))

  def test_generic_add_with_unplaced_named_tuple_and_tensor(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.StructType([[('a', tf.float32), ('b', tf.float32)],
                                      tf.float32]))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri](x)

    self.assertEqual(
        str(foo.type_signature),
        '(<<a=float32,b=float32>,float32> -> <a=float32,b=float32>)')

    self.assertEqual(
        foo([[1., 1.], 1.]), structure.Struct([('a', 2.), ('b', 2.)]))

  def test_generic_add_federated_named_tuple_by_tensor(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(
        computation_types.FederatedType([[('a', tf.float32),
                                          ('b', tf.float32)], tf.float32],
                                        placement_literals.CLIENTS))
    def foo(x):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x[0], x[1]])

    self.assertEqual(
        str(foo.type_signature),
        '({<<a=float32,b=float32>,float32>}@CLIENTS -> {<a=float32,b=float32>}@CLIENTS)'
    )

    self.assertEqual(
        foo([[[1., 1.], 1.]]), [structure.Struct([('a', 2.), ('b', 2.)])])
    self.assertEqual(
        foo([[[1., 1.], 1.], [[1., 2.], 2.], [[1., 4.], 4.]]), [
            structure.Struct([('a', 2.), ('b', 2.)]),
            structure.Struct([('a', 3.), ('b', 4.)]),
            structure.Struct([('a', 5.), ('b', 8.)])
        ])


if __name__ == '__main__':
  common_libs_test_utils.main()
