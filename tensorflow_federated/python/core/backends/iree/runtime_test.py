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

import tempfile

import iree.compiler.tf
import iree.runtime
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.iree import backend_info
from tensorflow_federated.python.core.backends.iree import computation_module
from tensorflow_federated.python.core.backends.iree import runtime
from tensorflow_federated.python.core.impl.types import computation_types


class RuntimeTest(tf.test.TestCase):

  def test_get_config_for_driver(self):
    c1 = runtime._get_default_config_for_driver('vulkan')
    self.assertIsInstance(c1, iree.runtime.Config)
    c2 = runtime._get_default_config_for_driver('vulkan')
    self.assertIs(c1, c2)
    c3 = runtime._get_default_config_for_driver('vmvx')
    self.assertIsInstance(c3, iree.runtime.Config)
    self.assertIsNot(c2, c3)

  def test_computation_callable(self):
    tf_module = tf.Module()
    fn = lambda x: x + 1.0
    sig = [tf.TensorSpec([], tf.float32)]
    tf_module.foo = tf.function(fn, input_signature=sig)
    with tempfile.TemporaryDirectory() as model_dir:
      save_options = tf.saved_model.SaveOptions(save_debug_info=True)
      tf.saved_model.save(tf_module, model_dir, options=save_options)
      iree_compiler_module = iree.compiler.tf.compile_saved_model(
          model_dir, import_only=True)
    my_computation_module = computation_module.ComputationModule(
        iree_compiler_module, 'foo',
        computation_types.FunctionType(tf.float32, tf.float32))
    computation_callable = runtime.ComputationCallable(
        my_computation_module, backend_info.VULKAN_SPIRV)
    self.assertTrue(callable(computation_callable))
    result = computation_callable(np.float32(5.0))
    self.assertEqual(result, 6.0)


if __name__ == '__main__':
  tf.test.main()
