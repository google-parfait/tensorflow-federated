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

from pyiree.compiler import tf as iree_compiler_tf
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.backends.iree import computation_module


class ComputationModuleTest(tf.test.TestCase):

  def test_module_class_with_add_one(self):
    tf_module = tf.Module()
    tf_module.foo = tf.function(
        lambda x: x + 1.0,
        input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    model_dir = '/tmp/foo'
    save_options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(tf_module, model_dir, options=save_options)
    iree_compiler_module = iree_compiler_tf.compile_saved_model(
        model_dir,
        import_only=True,
        exported_names=['foo'],
        target_backends=iree_compiler_tf.DEFAULT_TESTING_BACKENDS)
    my_computation_module = computation_module.ComputationModule(
        iree_compiler_module, 'foo',
        computation_types.FunctionType(tf.float32, tf.float32))
    self.assertIs(my_computation_module.compiler_module, iree_compiler_module)
    self.assertEqual(my_computation_module.function_name, 'foo')
    self.assertEqual(
        str(my_computation_module.type_signature), '(float32 -> float32)')


if __name__ == '__main__':
  tf.test.main()
