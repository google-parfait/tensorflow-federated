# Copyright 2020, The TensorFlow Federated Authors.
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

import re

import numpy as np
import tensorflow as tf

from iree.integrations.tensorflow.bindings.python.pyiree.tf import compiler as iree_compiler
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.iree import compiler
from tensorflow_federated.python.core.impl import computation_impl


class CompilerTest(tf.test.TestCase):

  def test_import_tf_comp_with_one_constant(self):

    @computations.tf_computation
    def comp():
      return 99.0

    _, mlir = self._import_compile_and_return_module_and_mlir(comp)
    self._assert_mlir_contains_pattern(mlir, [
        'func @fn() -> tensor<f32> SOMETHING {',
        '  %0 = xla_hlo.constant dense<9.900000e+01>',
        '  return %0',
        '}',
    ])

  def test_import_tf_comp_with_one_variable_constant(self):

    @computations.tf_computation
    def comp():
      return tf.Variable(99.0)

    _, mlir = self._import_compile_and_return_module_and_mlir(comp)
    self._assert_mlir_contains_pattern(mlir, [
        'func @fn() -> tensor<f32> SOMETHING {',
        '  %0 = flow.variable.address',
        '  %1 = xla_hlo.constant dense<9.900000e+01>',
        '  flow.variable.store.indirect %1, %0',
        '  %2 = flow.variable.load.indirect %0',
        '  return %2',
        '}',
    ])

  def test_import_tf_comp_with_add_one(self):

    @computations.tf_computation(tf.float32)
    def comp(x):
      return x + 1.0

    _, mlir = self._import_compile_and_return_module_and_mlir(comp)
    self._assert_mlir_contains_pattern(mlir, [
        'func @fn(%arg0: tensor<f32>) -> tensor<f32> SOMETHING {',
        '  %0 = xla_hlo.constant dense<1.000000e+00>',
        '  %1 = xla_hlo.add %arg0, %0',
        '  return %1',
        '}',
    ])

  def test_import_tf_comp_with_variable_add_one(self):

    @computations.tf_computation(tf.float32)
    def comp(x):
      v = tf.Variable(1.0)
      with tf.control_dependencies([v.initializer]):
        return tf.add(v, x)

    _, mlir = self._import_compile_and_return_module_and_mlir(comp)
    self._assert_mlir_contains_pattern(mlir, [
        'func @fn(%arg0: tensor<f32>) -> tensor<f32> SOMETHING {',
        '  %0 = flow.variable.address',
        '  %1 = xla_hlo.constant dense<1.000000e+00>',
        '  flow.variable.store.indirect %1, %0',
        '  %2 = flow.variable.load.indirect %0',
        '  %3 = xla_hlo.add %2, %arg0',
        '  return %3',
        '}',
    ])

  def test_import_tf_comp_fails_with_non_tf_comp(self):

    @computations.federated_computation
    def comp():
      return 10.0

    with self.assertRaises(TypeError):
      self._import_compile_and_return_module_and_mlir(comp)

  def test_import_tf_comp_fails_with_dataset_parameter(self):

    @computations.tf_computation(computation_types.SequenceType(tf.float32))
    def comp(x):
      return x.reduce(np.float32(0), lambda x, y: x + y)

    # TODO(b/153499219): Convert this into a test of successful compilation.
    with self.assertRaises(TypeError):
      self._import_compile_and_return_module_and_mlir(comp)

  def test_import_tf_comp_fails_with_dataset_result(self):

    @computations.tf_computation
    def comp():
      return tf.data.Dataset.range(10)

    # TODO(b/153499219): Convert this into a test of successful compilation.
    with self.assertRaises(TypeError):
      self._import_compile_and_return_module_and_mlir(comp)

  def test_import_tf_comp_fails_with_named_tuple_parameter(self):

    @computations.tf_computation(tf.float32, tf.float32)
    def comp(x, y):
      return tf.add(x, y)

    # TODO(b/153499219): Convert this into a test of successful compilation.
    with self.assertRaises(TypeError):
      self._import_compile_and_return_module_and_mlir(comp)

  def test_import_tf_comp_fails_with_named_tuple_result(self):

    @computations.tf_computation
    def comp():
      return 10.0, 20.0

    # TODO(b/153499219): Convert this into a test of successful compilation.
    with self.assertRaises(TypeError):
      self._import_compile_and_return_module_and_mlir(comp)

  def _import_compile_and_return_module_and_mlir(self, comp):
    """Testing helper that compiles `comp` and returns the compiler module.

    Args:
      comp: A computation created with `@computations.tf_computation`.

    Returns:
      A tuple consisting of the compiler module and MLIR.
    """
    comp_proto = computation_impl.ComputationImpl.get_proto(comp)
    comp_type = comp.type_signature
    module = compiler.import_tensorflow_computation(comp_proto, comp_type)
    self.assertIsInstance(module, iree_compiler.binding.CompilerModule)
    mlir = module.to_asm(large_element_limit=100)
    return module, mlir

  def _assert_mlir_contains_pattern(self, actual_mlir, expected_list):
    """Testing helper that verifies received MLIR against the expectations.

    Args:
      actual_mlir: The MLIR string to test.
      expected_list: An expected list of MLIR strings to look for in the order
        in which they are expected to apppear, potentially separated by any
        sequences of characters. Leading and trailing whitespaces are ignored.
        The text SOMETHING in the expected strings indicates a wildcard.
    """
    escaped_pattern = '.*'.join(
        re.escape(x.strip()).replace('SOMETHING', '.*') for x in expected_list)
    self.assertRegex(actual_mlir.replace('\n', ' '), escaped_pattern)


if __name__ == '__main__':
  tf.test.main()
