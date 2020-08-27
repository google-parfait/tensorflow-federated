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

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances


class TensorflowWrapperTest(test.TestCase):

  def test_invoke_with_lambda(self):
    foo = lambda x: x > 10
    foo = computation_wrapper_instances.tensorflow_wrapper(foo, tf.int32)
    self.assertEqual(foo.type_signature.compact_representation(),
                     '(int32 -> bool)')

  def test_invoke_with_polymorphic_lambda(self):
    foo = lambda x: x > 10
    foo = computation_wrapper_instances.tensorflow_wrapper(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.int32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(int32 -> bool)')
    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.float32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(float32 -> bool)')

  def test_invoke_with_no_arg_lambda(self):
    foo = lambda: 10
    foo = computation_wrapper_instances.tensorflow_wrapper(foo)
    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')

  def test_invoke_with_fn(self):

    def foo(x):
      return x > 10

    foo = computation_wrapper_instances.tensorflow_wrapper(foo, tf.int32)
    self.assertEqual(foo.type_signature.compact_representation(),
                     '(int32 -> bool)')

  def test_invoke_with_polymorphic_fn(self):

    def foo(x):
      return x > 10

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.int32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(int32 -> bool)')
    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.float32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(float32 -> bool)')

  def test_invoke_with_no_arg_fn(self):

    def foo():
      return 10

    foo = computation_wrapper_instances.tensorflow_wrapper(foo)
    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')

  def test_decorate_as_fn(self):

    @computation_wrapper_instances.tensorflow_wrapper(tf.int32)
    def foo(x):
      return x > 10

    self.assertEqual(foo.type_signature.compact_representation(),
                     '(int32 -> bool)')

  def test_decorate_as_polymorphic_fn(self):

    @computation_wrapper_instances.tensorflow_wrapper
    def foo(x):
      return x > 10

    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.int32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(int32 -> bool)')
    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.float32))
    self.assertEqual(concrete_fn.type_signature.compact_representation(),
                     '(float32 -> bool)')

  def test_decorate_as_no_arg_fn(self):

    @computation_wrapper_instances.tensorflow_wrapper
    def foo():
      return 10

    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')

  def test_fails_with_bad_types(self):
    function = computation_types.FunctionType(
        None, computation_types.TensorType(tf.int32))
    federated = computation_types.FederatedType(tf.int32,
                                                placement_literals.CLIENTS)
    tuple_on_function = computation_types.StructType([federated, function])

    def foo(x):  # pylint: disable=unused-variable
      del x  # Unused.

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type {int32}@CLIENTS'):
      computation_wrapper_instances.tensorflow_wrapper(foo, federated)

    # pylint: disable=anomalous-backslash-in-string
    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type \( -> int32\)'):
      computation_wrapper_instances.tensorflow_wrapper(foo, function)

    with self.assertRaisesRegex(
        TypeError, r'you have attempted to create one with the type placement'):
      computation_wrapper_instances.tensorflow_wrapper(
          foo, computation_types.PlacementType())

    with self.assertRaisesRegex(
        TypeError, r'you have attempted to create one with the type T'):
      computation_wrapper_instances.tensorflow_wrapper(
          foo, computation_types.AbstractType('T'))

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type <{int32}@CLIENTS,\( '
        '-> int32\)>'):
      computation_wrapper_instances.tensorflow_wrapper(foo, tuple_on_function)
    # pylint: enable=anomalous-backslash-in-string


class FederatedComputationWrapperTest(test.TestCase):

  def test_federated_computation_wrapper(self):

    @computation_wrapper_instances.federated_computation_wrapper(
        (computation_types.FunctionType(tf.int32, tf.int32), tf.int32))
    def foo(f, x):
      return f(f(x))

    self.assertIsInstance(foo, computation_impl.ComputationImpl)
    self.assertEqual(
        str(foo.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)')

    self.assertEqual(
        str(foo.to_building_block()),
        '(FEDERATED_arg -> (let fc_FEDERATED_symbol_0=FEDERATED_arg.f(FEDERATED_arg.x),fc_FEDERATED_symbol_1=FEDERATED_arg.f(fc_FEDERATED_symbol_0) in fc_FEDERATED_symbol_1))'
    )


class ToComputationImplTest(test.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      computation_wrapper_instances.building_block_to_computation(None)

  def test_converts_building_block_to_computation(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    computation_impl_lambda = computation_wrapper_instances.building_block_to_computation(
        lam)
    self.assertIsInstance(computation_impl_lambda,
                          computation_impl.ComputationImpl)


if __name__ == '__main__':
  test.main()
