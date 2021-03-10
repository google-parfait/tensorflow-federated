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

import warnings

import tensorflow as tf

from tensorflow_federated.python.simulation import file_per_user_client_data

# TODO(b/182305417): Delete this once the full deprecation period has passed.


class FilePerUserClientDataTest(tf.test.TestCase):

  def test_deprecation_warning_raised_on_init(self):
    ids_to_files = {'client_id': 'file'}
    dataset_fn = lambda x: tf.data.Dataset.range(5)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      file_per_user_client_data.FilePerUserClientData(ids_to_files, dataset_fn)
      self.assertNotEmpty(w)
      self.assertEqual(w[0].category, DeprecationWarning)
      self.assertRegex(
          str(w[0].message),
          'tff.simulation.FilePerUserClientData is deprecated')


if __name__ == '__main__':
  tf.test.main()
