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

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.utils import computation_utils


class ComputationUtilsTest(test.TestCase):

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
    with self.assertRaisesRegex(TypeError, 'state must be a structure'):
      computation_utils.update_state((1, 2, 3), a=8)
    with self.assertRaisesRegex(TypeError, 'state must be a structure'):
      computation_utils.update_state([1, 2, 3], a=8)
    with self.assertRaisesRegex(KeyError, 'does not contain a field'):
      computation_utils.update_state({'z': 1}, a=8)


if __name__ == '__main__':
  test.main()
