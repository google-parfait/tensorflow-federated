# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections

import attr

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.utils import computation_utils


# Convenience alias.
Struct = structure.Struct


class ComputationUtilsTest(test_case.TestCase):

  def test_update_state_tff_struct(self):
    with self.subTest('fully_named'):
      state = Struct([('a', 1), ('b', 2), ('c', 3)])
      state = computation_utils.update_state(state, c=7)
      self.assertEqual(state, Struct([('a', 1), ('b', 2), ('c', 7)]))
      state = computation_utils.update_state(state, a=8)
      self.assertEqual(state, Struct([('a', 8), ('b', 2), ('c', 7)]))
    with self.subTest('partially_named'):
      state = Struct([(None, 1), ('b', 2), (None, 3)])
      state = computation_utils.update_state(state, b=7)
      self.assertEqual(state, Struct([(None, 1), ('b', 7), (None, 3)]))
      with self.assertRaises(KeyError):
        computation_utils.update_state(state, a=8)
    with self.subTest('nested'):
      state = Struct([('a', {'a1': 1, 'a2': 2}), ('b', 2), ('c', 3)])
      state = computation_utils.update_state(state, a=7)
      self.assertEqual(state, Struct([('a', 7), ('b', 2), ('c', 3)]))
      state = computation_utils.update_state(state, a={'foo': 1, 'bar': 2})
      self.assertEqual(
          state, Struct([('a', {
              'foo': 1,
              'bar': 2
          }), ('b', 2), ('c', 3)]))
    with self.subTest('unnamed'):
      state = Struct((None, i) for i in range(3))
      with self.assertRaises(KeyError):
        computation_utils.update_state(state, a=1)
      with self.assertRaises(KeyError):
        computation_utils.update_state(state, b=1)

  def test_update_state_namedtuple(self):
    my_tuple_type = collections.namedtuple('my_tuple_type', 'a b c')
    state = my_tuple_type(1, 2, 3)
    state2 = computation_utils.update_state(state, c=7)
    self.assertEqual(state2, my_tuple_type(1, 2, 7))
    state3 = computation_utils.update_state(state2, a=8)
    self.assertEqual(state3, my_tuple_type(8, 2, 7))

  def test_update_state_dict(self):
    state = collections.OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    state2 = computation_utils.update_state(state, c=7)
    self.assertEqual(state2, {'a': 1, 'b': 2, 'c': 7})
    state3 = computation_utils.update_state(state2, a=8)
    self.assertEqual(state3, {'a': 8, 'b': 2, 'c': 7})

  def test_update_state_ordereddict(self):
    state = collections.OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    state2 = computation_utils.update_state(state, c=7)
    self.assertEqual(state2,
                     collections.OrderedDict([('a', 1), ('b', 2), ('c', 7)]))
    state3 = computation_utils.update_state(state2, a=8)
    self.assertEqual(state3,
                     collections.OrderedDict([('a', 8), ('b', 2), ('c', 7)]))

  def test_update_state_attrs(self):

    @attr.s
    class TestAttrsClass(object):
      a = attr.ib()
      b = attr.ib()
      c = attr.ib()

    state = TestAttrsClass(1, 2, 3)
    state2 = computation_utils.update_state(state, c=7)
    self.assertEqual(state2, TestAttrsClass(1, 2, 7))
    state3 = computation_utils.update_state(state2, a=8)
    self.assertEqual(state3, TestAttrsClass(8, 2, 7))

  def test_update_state_fails(self):
    with self.assertRaisesRegex(TypeError, '`structure` must be a structure'):
      computation_utils.update_state((1, 2, 3), a=8)
    with self.assertRaisesRegex(TypeError, '`structure` must be a structure'):
      computation_utils.update_state([1, 2, 3], a=8)
    with self.assertRaisesRegex(KeyError, 'does not contain a field'):
      computation_utils.update_state({'z': 1}, a=8)


if __name__ == '__main__':
  test_case.main()
