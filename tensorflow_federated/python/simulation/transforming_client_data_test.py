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
from tensorflow_federated.python.simulation import transforming_client_data
from tensorflow_federated.python.simulation.datasets import from_tensor_slices_client_data

# TODO(b/182305417): Delete this once the full deprecation period has passed.

TEST_DATA = {
    'CLIENT A': {
        'x': [[1, 2], [3, 4], [5, 6]],
    },
}


def _test_transform_cons(raw_client_id, index):
  del raw_client_id

  def fn(data):
    data['x'] = data['x'] + 10 * index
    return data

  return fn


class TransformingClientDataTest(test_case.TestCase):

  def test_client_ids_property(self):
    client_data = from_tensor_slices_client_data.FromTensorSlicesClientData(
        TEST_DATA)

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      transforming_client_data.TransformingClientData(client_data,
                                                      _test_transform_cons)
      self.assertNotEmpty(w)
      self.assertEqual(w[0].category, DeprecationWarning)
      self.assertRegex(
          str(w[0].message),
          'tff.simulation.TransformingClientData is deprecated')


if __name__ == '__main__':
  test_case.main()
