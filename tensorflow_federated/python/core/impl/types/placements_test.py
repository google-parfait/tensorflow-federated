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

from absl.testing import absltest

from tensorflow_federated.python.core.impl.types import placements


class PlacementLiteralsTest(absltest.TestCase):

  def test_something(self):
    self.assertNotEqual(str(placements.CLIENTS), str(placements.SERVER))
    for literal in [placements.CLIENTS, placements.SERVER]:
      self.assertIs(placements.uri_to_placement_literal(literal.uri), literal)

  def test_comparators_and_hashing(self):
    self.assertEqual(placements.CLIENTS, placements.CLIENTS)
    self.assertNotEqual(placements.CLIENTS, placements.SERVER)
    self.assertEqual(hash(placements.CLIENTS), hash(placements.CLIENTS))
    self.assertNotEqual(hash(placements.CLIENTS), hash(placements.SERVER))
    foo = {placements.CLIENTS: 10, placements.SERVER: 20}
    self.assertEqual(foo[placements.CLIENTS], 10)
    self.assertEqual(foo[placements.SERVER], 20)

  def test_comparison_to_none(self):
    self.assertNotEqual(placements.CLIENTS, None)
    self.assertNotEqual(placements.SERVER, None)
    self.assertNotEqual(None, placements.CLIENTS)
    self.assertNotEqual(None, placements.SERVER)


if __name__ == '__main__':
  absltest.main()
