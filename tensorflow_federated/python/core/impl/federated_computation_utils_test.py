# Lint as: python3
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
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl.utils import function_utils


class ComputationBuildingUtilsTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.parameters(
      (lambda f, x: f(f(x)),
       [('f', computation_types.FunctionType(tf.int32, tf.int32)),
        ('x', tf.int32)],
       '(FEDERATED_foo -> FEDERATED_foo.f(FEDERATED_foo.f(FEDERATED_foo.x)))'),
      (lambda f, g, x: f(g(x)),
       [('f', computation_types.FunctionType(tf.int32, tf.int32)),
        ('g', computation_types.FunctionType(tf.int32, tf.int32)),
        ('x', tf.int32)],
       '(FEDERATED_foo -> FEDERATED_foo.f(FEDERATED_foo.g(FEDERATED_foo.x)))'),
      (lambda x: (x[1], x[0]),
       (tf.int32, tf.int32),
       '(FEDERATED_foo -> <FEDERATED_foo[1],FEDERATED_foo[0]>)'),
      (lambda: 'stuff', None, 'comp#'))
  # pyformat: enable
  def test_zero_or_one_arg_fn_to_building_block(self, fn, parameter_type,
                                                fn_str):
    parameter_name = 'foo'
    parameter_type = computation_types.to_type(parameter_type)
    fn = function_utils.wrap_as_zero_or_one_arg_callable(fn, parameter_type)
    result = federated_computation_utils.zero_or_one_arg_fn_to_building_block(
        fn, parameter_name, parameter_type, context_stack_impl.context_stack)
    self.assertStartsWith(str(result), fn_str)


if __name__ == '__main__':
  absltest.main()
