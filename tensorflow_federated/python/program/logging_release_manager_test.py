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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.program import logging_release_manager
from tensorflow_federated.python.program import test_utils


class LoggingReleaseManagerTest(parameterized.TestCase, tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, None),
      ('bool', True, True),
      ('int', 1, 1),
      ('str', 'a', 'a'),
      ('list', [True, 1, 'a'], [True, 1, 'a']),
      ('list_empty', [], []),
      ('list_nested', [[True, 1], ['a']], [[True, 1], ['a']]),
      ('dict', {'a': True, 'b': 1, 'c': 'a'}, {'a': True, 'b': 1, 'c': 'a'}),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}}),
      ('attr',
       test_utils.TestAttrObject1(True, 1),
       test_utils.TestAttrObject1(True, 1)),
      ('attr_nested',
       {'a': [test_utils.TestAttrObject1(True, 1)],
        'b': test_utils.TestAttrObject2('a')},
       {'a': [test_utils.TestAttrObject1(True, 1)],
        'b': test_utils.TestAttrObject2('a')}),
      ('tensor_int', tf.constant(1), tf.constant(1)),
      ('tensor_str', tf.constant('a'), tf.constant('a')),
      ('tensor_2d', tf.ones((2, 3)), tf.ones((2, 3))),
      ('tensor_nested',
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]},
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]}),
      ('numpy_int', np.int32(1), np.int32(1)),
      ('numpy_2d', np.ones((2, 3)), np.ones((2, 3))),
      ('numpy_nested',
       {'a': [np.bool(True), np.int32(1)], 'b': [np.str_('a')]},
       {'a': [np.bool(True), np.int32(1)], 'b': [np.str_('a')]}),
      ('server_array_reference', test_utils.TestServerArrayReference(1), 1),
      ('server_array_reference_nested',
       {'a': [test_utils.TestServerArrayReference(True),
              test_utils.TestServerArrayReference(1)],
        'b': [test_utils.TestServerArrayReference('a')]},
       {'a': [True, 1], 'b': ['a']}),
      ('materialized_values_and_value_references',
       [1, test_utils.TestServerArrayReference(2)],
       [1, 2]),
  )
  # pyformat: enable
  def test_release_logs_value(self, value, expected_value):
    release_mngr = logging_release_manager.LoggingReleaseManager()

    with mock.patch('absl.logging.info') as mock_info:
      release_mngr.release(value)

      mock_info.assert_called_once()
      call = mock_info.mock_calls[0]
      _, args, _ = call
      _, actual_value = args
      self.assertAllEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_release_logs_key(self, key):
    release_mngr = logging_release_manager.LoggingReleaseManager()

    with mock.patch('absl.logging.info') as mock_info:
      release_mngr.release(1, key)

      mock_info.assert_called_once()
      call = mock_info.mock_calls[0]
      _, args, _ = call
      _, actual_key, actual_value = args
      self.assertEqual(actual_key, key)
      self.assertEqual(actual_value, 1)

if __name__ == '__main__':
  absltest.main()
