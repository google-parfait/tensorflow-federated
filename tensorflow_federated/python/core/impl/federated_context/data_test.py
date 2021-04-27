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

import tensorflow as tf

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.federated_context import data
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types


class DataTest(test_case.TestCase):

  def test_data(self):
    val = data.data('foo://bar', computation_types.SequenceType(tf.int32))
    self.assertIsInstance(val, value_impl.Value)
    self.assertEqual(str(val.type_signature), 'int32*')
    self.assertIsInstance(val.comp, building_blocks.Data)
    self.assertEqual(val.comp.uri, 'foo://bar')


if __name__ == '__main__':
  test_case.main()
