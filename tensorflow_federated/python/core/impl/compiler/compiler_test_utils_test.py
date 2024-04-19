# Copyright 2022, The TensorFlow Federated Authors.
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

import collections

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.core.impl.compiler import building_block_test_utils
from tensorflow_federated.python.core.impl.compiler import compiler_test_utils


class CheckCompiledComputationsTest(parameterized.TestCase):

  def test_check_computations_succeeds(self):
    computation_1 = building_block_test_utils.create_nested_syntax_tree()
    computation_2 = building_block_test_utils.create_nested_syntax_tree()
    computations = collections.OrderedDict(
        computation_1=computation_1, computation_2=computation_2
    )
    compiler_test_utils.check_computations(
        'test_check_computations.expected', computations
    )

  @parameterized.named_parameters([
      (
          'invalid_filename',
          1.0,
          collections.OrderedDict(
              computation=building_block_test_utils.create_nested_syntax_tree()
          ),
      ),
      (
          'invalid_computations',
          'test_check_computations.expected',
          [building_block_test_utils.create_nested_syntax_tree()],
      ),
      (
          'invalid_computation',
          'test_check_computations.expected',
          collections.OrderedDict(computation=1.0),
      ),
  ])
  def test_check_computations_raises_type_error(self, filename, computations):
    with self.assertRaises(TypeError):
      compiler_test_utils.check_computations(filename, computations)


if __name__ == '__main__':
  absltest.main()
