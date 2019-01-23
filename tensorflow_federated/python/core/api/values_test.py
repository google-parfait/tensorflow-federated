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
"""Tests for values.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.api import values


class ValuesTest(absltest.TestCase):

  # NOTE: No need to test all supported types, as those are already tested in
  # the test of the underlying implementation (`value_impl_test.py`).
  def test_to_value_with_int_constant(self):
    val = values.to_value(10)
    self.assertIsInstance(val, value_base.Value)
    self.assertEqual(str(val.type_signature), 'int32')


if __name__ == '__main__':
  absltest.main()
