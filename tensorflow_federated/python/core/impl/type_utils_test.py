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

import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.impl import type_utils


class TypeUtilsTest(test.TestCase, parameterized.TestCase):

  def test_to_canonical_value_with_none(self):
    self.assertEqual(type_utils.to_canonical_value(None), None)

  def test_to_canonical_value_with_int(self):
    self.assertEqual(type_utils.to_canonical_value(1), 1)

  def test_to_canonical_value_with_float(self):
    self.assertEqual(type_utils.to_canonical_value(1.0), 1.0)

  def test_to_canonical_value_with_bool(self):
    self.assertEqual(type_utils.to_canonical_value(True), True)
    self.assertEqual(type_utils.to_canonical_value(False), False)

  def test_to_canonical_value_with_string(self):
    self.assertEqual(type_utils.to_canonical_value('a'), 'a')

  def test_to_canonical_value_with_list_of_ints(self):
    self.assertEqual(type_utils.to_canonical_value([1, 2, 3]), [1, 2, 3])

  def test_to_canonical_value_with_list_of_floats(self):
    self.assertEqual(
        type_utils.to_canonical_value([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

  def test_to_canonical_value_with_list_of_bools(self):
    self.assertEqual(
        type_utils.to_canonical_value([True, False]), [True, False])

  def test_to_canonical_value_with_list_of_strings(self):
    self.assertEqual(
        type_utils.to_canonical_value(['a', 'b', 'c']), ['a', 'b', 'c'])

  def test_to_canonical_value_with_list_of_dict(self):
    self.assertEqual(
        type_utils.to_canonical_value([{
            'a': 1,
            'b': 0.1,
        }]), [anonymous_tuple.AnonymousTuple([
            ('a', 1),
            ('b', 0.1),
        ])])

  def test_to_canonical_value_with_list_of_ordered_dict(self):
    self.assertEqual(
        type_utils.to_canonical_value(
            [collections.OrderedDict([
                ('a', 1),
                ('b', 0.1),
            ])]), [anonymous_tuple.AnonymousTuple([
                ('a', 1),
                ('b', 0.1),
            ])])

  def test_to_canonical_value_with_dict(self):
    self.assertEqual(
        type_utils.to_canonical_value({
            'a': 1,
            'b': 0.1,
        }), anonymous_tuple.AnonymousTuple([
            ('a', 1),
            ('b', 0.1),
        ]))
    self.assertEqual(
        type_utils.to_canonical_value({
            'b': 0.1,
            'a': 1,
        }), anonymous_tuple.AnonymousTuple([
            ('a', 1),
            ('b', 0.1),
        ]))

  def test_to_canonical_value_with_ordered_dict(self):
    self.assertEqual(
        type_utils.to_canonical_value(
            collections.OrderedDict([
                ('a', 1),
                ('b', 0.1),
            ])), anonymous_tuple.AnonymousTuple([
                ('a', 1),
                ('b', 0.1),
            ]))
    self.assertEqual(
        type_utils.to_canonical_value(
            collections.OrderedDict([
                ('b', 0.1),
                ('a', 1),
            ])), anonymous_tuple.AnonymousTuple([
                ('b', 0.1),
                ('a', 1),
            ]))


if __name__ == '__main__':
  tf.test.main()
