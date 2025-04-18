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

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
import numpy as np
from tensorflow_federated.python.core.impl.executors import executor_utils


class TypeUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'buiding_block_and_type_spec',
          federated_language.framework.create_identity(
              federated_language.TensorType(np.int32)
          ),
          federated_language.FunctionType(np.int32, np.int32),
          federated_language.FunctionType(np.int32, np.int32),
      ),
      (
          'buiding_block_and_none',
          federated_language.framework.create_identity(
              federated_language.TensorType(np.int32)
          ),
          None,
          federated_language.FunctionType(np.int32, np.int32),
      ),
      (
          'int_and_type_spec',
          10,
          federated_language.TensorType(np.int32),
          federated_language.TensorType(np.int32),
      ),
  ])
  def test_reconcile_value_with_type_spec_returns_type(
      self, value, type_spec, expected_type
  ):
    actual_type = executor_utils.reconcile_value_with_type_spec(
        value, type_spec
    )
    self.assertEqual(actual_type, expected_type)

  @parameterized.named_parameters([
      (
          'building_block_and_bad_type_spec',
          federated_language.framework.create_identity(
              federated_language.TensorType(np.int32)
          ),
          federated_language.TensorType(np.int32),
      ),
      ('int_and_none', 10, None),
  ])
  def test_reconcile_value_with_type_spec_raises_type_error(
      self, value, type_spec
  ):
    with self.assertRaises(TypeError):
      executor_utils.reconcile_value_with_type_spec(value, type_spec)

  @parameterized.named_parameters([
      (
          'value_type_and_type_spec',
          federated_language.TensorType(np.int32),
          federated_language.TensorType(np.int32),
          federated_language.TensorType(np.int32),
      ),
      (
          'value_type_and_none',
          federated_language.TensorType(np.int32),
          None,
          federated_language.TensorType(np.int32),
      ),
  ])
  def test_reconcile_value_type_with_type_spec_returns_type(
      self, value_type, type_spec, expected_type
  ):
    actual_type = executor_utils.reconcile_value_type_with_type_spec(
        value_type, type_spec
    )
    self.assertEqual(actual_type, expected_type)

  def test_reconcile_value_type_with_type_spec_raises_type_error_value_type_and_bad_type_spec(
      self,
  ):
    value_type = federated_language.TensorType(np.int32)
    type_spec = federated_language.TensorType(np.str_)

    with self.assertRaises(TypeError):
      executor_utils.reconcile_value_type_with_type_spec(value_type, type_spec)


if __name__ == '__main__':
  absltest.main()
