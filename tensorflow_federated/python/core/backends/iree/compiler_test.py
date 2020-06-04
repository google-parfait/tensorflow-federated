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

from iree.bindings.python.pyiree import rt as iree_runtime
from iree.integrations.tensorflow.bindings.python.pyiree.tf import compiler as iree_compiler
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.iree import compiler
from tensorflow_federated.python.core.impl import computation_impl


# The config must be shared across contexts, otherwise the runtime crashes.
# TODO(b/153499219): Possibly remove this once IREE moves to the singleton
# pattern to ensure there is never more than one config.
_IREE_RUNTIME_CONFIG = iree_runtime.Config('vulkan')


class CompilerTest(tf.test.TestCase):

  def test_import_tf_comp_with_one_constant(self):

    @computations.tf_computation
    def comp():
      return 99.0

    module, mlir = self._import_compile_and_return_module_and_mlir(comp)
    self._assert_mlir_contains_pattern(mlir, [
        'func @fn() -> tensor<f32> SOMETHING {',
        '  %0 = xla_hlo.constant dense<9.900000e+01>',
        '  return %0',
        '}',
    ])
    result = self._run_imported_module_on_args(module)
    self.assertEqual(result, 99.0)

  def test_import_tf_comp_with_one_variable_constant(self):

    @computations.tf_computation
    def comp():
      return tf.Variable(99.0)

    module, mlir = self._import_compile_and_return_module_and_mlir(comp)
    self._assert_mlir_contains_pattern(mlir, [
        'func @fn() -> tensor<f32> SOMETHING {',
        '  %0 = flow.variable.address',
        '  %1 = xla_hlo.constant dense<9.900000e+01>',
        '  flow.variable.store.indirect %1, %0',
        '  %2 = flow.variable.load.indirect %0',
        '  return %2',
        '}',
    ])
    result = self._run_imported_module_on_args(module)
    self.assertEqual(result, 99.0)

  def test_import_tf_comp_with_add_one(self):

    @computations.tf_computation(tf.float32)
    def comp(x):
      return x + 1.0

    module, mlir = self._import_compile_and_return_module_and_mlir(comp)
    self._assert_mlir_contains_pattern(mlir, [
        'func @fn(%arg0: tensor<f32>) -> tensor<f32> SOMETHING {',
        '  %0 = xla_hlo.constant dense<1.000000e+00>',
        '  %1 = xla_hlo.add %arg0, %0',
        '  return %1',
        '}',
    ])
    result = self._run_imported_module_on_args(module, np.float32(5.0))
    self.assertEqual(result, 6.0)

  def test_import_tf_comp_with_variable_add_one(self):

    @computations.tf_computation(tf.float32)
    def comp(x):
      v = tf.Variable(1.0)
      with tf.control_dependencies([v.initializer]):
        return tf.add(v, x)

    module, mlir = self._import_compile_and_return_module_and_mlir(comp)
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
    result = self._run_imported_module_on_args(module, np.float32(5.0))
    self.assertEqual(result, 6.0)

  def test_import_tf_comp_with_variable_assign_add_one(self):

    @computations.tf_computation(tf.float32)
    def comp(x):
      v = tf.Variable(1.0)
      with tf.control_dependencies([v.initializer]):
        with tf.control_dependencies([v.assign_add(x)]):
          return tf.identity(v)

    module, mlir = self._import_compile_and_return_module_and_mlir(comp)

    # TODO(b/153499219): Introduce the concept of local variables, so that code
    # like what's in this section below can be dramatically simplified.
    self._assert_mlir_contains_pattern(mlir, [
        'flow.variable SOMETHING mutable dense<1.000000e+00> : tensor<f32>',
        'func @fn(%arg0: tensor<f32>) -> tensor<f32> SOMETHING {',
        '  %0 = flow.variable.address',
        '  %1 = xla_hlo.constant dense<1.000000e+00>',
        '  flow.variable.store.indirect %1, %0',
        '  %2 = flow.variable.load.indirect %0',
        '  %3 = xla_hlo.add %2, %arg0',
        '  flow.variable.store.indirect %3, %0',
        '  %4 = flow.variable.load.indirect %0',
        '  return %4',
        '}',
    ])
    result = self._run_imported_module_on_args(module, np.float32(5.0))
    self.assertEqual(result, 6.0)

  def test_import_tf_comp_with_while_loop(self):

    @computations.tf_computation(tf.float32)
    def comp(x):
      # An example of a loop with variables that computes 2^x by counting from
      # x down to 0, and doubling the result in each iteration.
      a = tf.Variable(0.0)
      b = tf.Variable(1.0)
      with tf.control_dependencies([a.initializer, b.initializer]):
        with tf.control_dependencies([a.assign(x)]):
          cond_fn = lambda a, b: a > 0
          body_fn = lambda a, b: (a - 1.0, b * 2.0)
          return tf.while_loop(cond_fn, body_fn, (a, b))[1]

    _, mlir = self._import_compile_and_return_module_and_mlir(comp)

    # Not checking the full MLIR in the long generated body, just that we can
    # successfully ingest TF code containing a while loop here, end-to-end. We
    # need some form of looping support in lieu of `tf.data.Dataset.reduce()`.
    self._assert_mlir_contains_pattern(
        mlir, ['func @fn(%arg0: tensor<f32>) -> tensor<f32>'])

    # TODO(b/153499219): Add the runtime test after fixing compilation errors,
    # and/or switch to vmla as the backend for this test, as vulkan has issues
    # with Boolean types.
    # * [ERROR]: SPIRV type conversion failed: 'memref<i1>'
    # * [ERROR]: failed to legalize operation 'iree.placeholder'
    # * [ERROR]: failed to run translation of source executable to target
    #            executable for backend vulkan.

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

  def _run_imported_module_on_args(self, module, *args):
    """Testing helper that runs the compiled module on a given set of args.

    Args:
      module: The module from `_import_compile_and_return_module_and_mlir()`.
      *args: Arguments for invocation.

    Returns:
      The result of invocation on IREE.
    """
    flatbuffer_blob = module.compile(target_backends=['vulkan-spirv'])
    vm_module = iree_runtime.VmModule.from_flatbuffer(flatbuffer_blob)
    context = iree_runtime.SystemContext(config=_IREE_RUNTIME_CONFIG)
    context.add_module(vm_module)
    # TODO(b/153499219): Can we possibly name the modules somehow differently?
    # This may not matter if we spawn a separate context for each, but it will
    # matter eventually. Right now, this comes from the implicit "module {}"
    # that wraps anything parsed from ASM that lacks an explicit module
    # declaration. Possibly manually surround with "module @myName { ... }" in
    # the compiler helpers.
    return context.modules.module.fn(*args)


if __name__ == '__main__':
  tf.test.main()
