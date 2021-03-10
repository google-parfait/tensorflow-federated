# Copyright 2019, The TensorFlow Federated Authors.
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

import warnings

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.simulation import from_tensor_slices_client_data

# TODO(b/182305417): Delete this once the full deprecation period has passed.


class FromTensorSlicesClientDataTest(test_case.TestCase):

  def test_deprecation_warning_raised_on_init(self):
    tensor_slices_dict = {'a': [1, 2, 3], 'b': [4, 5]}
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      from_tensor_slices_client_data.FromTensorSlicesClientData(
          tensor_slices_dict)
      self.assertNotEmpty(w)
      self.assertEqual(w[0].category, DeprecationWarning)
      self.assertRegex(
          str(w[0].message),
          'tff.simulation.FromTensorSlicesClientData is deprecated')


if __name__ == '__main__':
  test_case.main()
