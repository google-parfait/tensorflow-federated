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

from absl.testing import absltest
from tensorflow_federated.python.core.backends.iree import backend_info


class BackendInfoTest(absltest.TestCase):

  def test_vulkan_spirv(self):
    self.assertIsInstance(backend_info.VULKAN_SPIRV, backend_info.BackendInfo)
    self.assertEqual(backend_info.VULKAN_SPIRV.driver_name, 'vulkan')
    self.assertEqual(backend_info.VULKAN_SPIRV.target_name, 'vulkan-spirv')

  def test_vmla(self):
    self.assertIsInstance(backend_info.VMLA, backend_info.BackendInfo)
    self.assertEqual(backend_info.VMLA.driver_name, 'vmla')
    self.assertEqual(backend_info.VMLA.target_name, 'vmla')


if __name__ == '__main__':
  absltest.main()
