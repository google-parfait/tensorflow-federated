# Lint as: python3
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

from tensorflow_federated.python.core.impl.types import placement_literals


class PlacementLiteralsTest(absltest.TestCase):

  def test_something(self):
    self.assertNotEqual(
        str(placement_literals.CLIENTS), str(placement_literals.SERVER))
    for literal in [placement_literals.CLIENTS, placement_literals.SERVER]:
      self.assertIs(
          placement_literals.uri_to_placement_literal(literal.uri), literal)

  def test_comparators_and_hashing(self):
    self.assertEqual(placement_literals.CLIENTS, placement_literals.CLIENTS)
    self.assertNotEqual(placement_literals.CLIENTS, placement_literals.SERVER)
    self.assertEqual(
        hash(placement_literals.CLIENTS), hash(placement_literals.CLIENTS))
    self.assertNotEqual(
        hash(placement_literals.CLIENTS), hash(placement_literals.SERVER))
    foo = {placement_literals.CLIENTS: 10, placement_literals.SERVER: 20}
    self.assertEqual(foo[placement_literals.CLIENTS], 10)
    self.assertEqual(foo[placement_literals.SERVER], 20)


if __name__ == '__main__':
  absltest.main()
