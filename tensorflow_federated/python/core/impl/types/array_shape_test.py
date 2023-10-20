# Copyright 2023, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.impl.types import array_shape


class ArrayShapeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('fully_defined', (1, 2), True),
      ('partially_defined', (1, None), False),
      ('unknown', None, False),
      ('scalar', (), True),
  )
  def test_is_shape_fully_defined(self, shape, expected_result):
    actual_result = array_shape.is_shape_fully_defined(shape)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('fully_defined', (1, 2), 3),
      ('partially_defined', (1, None), None),
      ('unknown', None, None),
      ('scalar', (), 1),
  )
  def num_elements_in_shape(self, shape, expected_result):
    actual_result = array_shape.num_elements_in_shape(shape)
    self.assertEqual(actual_result, expected_result)


if __name__ == '__main__':
  absltest.main()
