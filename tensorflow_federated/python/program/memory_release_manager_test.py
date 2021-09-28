# Copyright 2021, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.program import memory_release_manager


class MemoryReleaseManagerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('none_0', None, 0),
      ('none_1', None, 1),
      ('int_0', 1, 0),
      ('int_1', 1, 1),
      ('list_0', [1, 2, 3], 0),
      ('list_1', [1, 2, 3], 1),
  )
  def test_release_saves_value_and_key(self, value, key):
    memory_release_mngr = memory_release_manager.MemoryReleaseManager()

    memory_release_mngr.release(value, key)

    self.assertLen(memory_release_mngr._values, 1)
    self.assertIn(key, memory_release_mngr._values)
    self.assertEqual(memory_release_mngr._values[key], value)

  @parameterized.named_parameters(
      ('none', None),
      ('int', 1),
      ('bool', True),
      ('str', 'a'),
      ('tuple', ()),
  )
  def test_release_does_not_raise_with_key(self, key):
    memory_release_mngr = memory_release_manager.MemoryReleaseManager()

    try:
      memory_release_mngr.release(1, key)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('list', []),
      ('dict', {}),
      ('orderd_dict', collections.OrderedDict()),
  )
  def test_release_raises_type_error_with_key(self, key):
    memory_release_mngr = memory_release_manager.MemoryReleaseManager()

    with self.assertRaises(TypeError):
      memory_release_mngr.release(1, key)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('10', 10),
  )
  def test_get_values_with_saved_values(self, count):
    memory_release_mngr = memory_release_manager.MemoryReleaseManager()
    for i in range(count):
      memory_release_mngr._values[i] = i * 10

    values = memory_release_mngr.values()

    self.assertEqual(values, {i: i * 10 for i in range(count)})

  def test_get_values_returns_copy(self):
    memory_release_mngr = memory_release_manager.MemoryReleaseManager()

    values = memory_release_mngr.values()
    self.assertEmpty(values)

    values[1] = 1

    values = memory_release_mngr.values()
    self.assertEmpty(values)


if __name__ == '__main__':
  absltest.main()
