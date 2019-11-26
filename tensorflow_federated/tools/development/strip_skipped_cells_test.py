# Lint as: python3
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

from tensorflow_federated.tools.development.strip_skipped_cells import SKIP_ANNOTATION
from tensorflow_federated.tools.development.strip_skipped_cells import strip_json


class StripJsonTest(absltest.TestCase):

  def assertEqualStripped(self, expected, input_value):
    self.assertEqual(expected, strip_json(input_value))

  def assertIdentity(self, v):
    self.assertEqualStripped(v, v)

  def test_simple_values_identity(self):
    for v in [42, 'foo', True, {}, []]:
      self.assertIdentity(v)

  def test_list_without_annotation_identity(self):
    self.assertIdentity(['foo', 42, True, {7: False}])

  def test_object_without_annotation_identity(self):
    self.assertIdentity({'foo': 42, True: [5]})

  def test_empties_list_with_annotation(self):
    testcases = [
        ([], [SKIP_ANNOTATION]),
        ([], [1, SKIP_ANNOTATION, 1]),
        ([1, []], [1, [SKIP_ANNOTATION]]),
        ({
            'a': 1,
            'b': []
        }, {
            'a': 1,
            'b': [SKIP_ANNOTATION]
        }),
    ]
    for (a, b) in testcases:
      self.assertEqualStripped(a, b)


if __name__ == '__main__':
  absltest.main()
