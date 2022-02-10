# Copyright 2022, The TensorFlow Federated Authors.
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
# limitations under the License.
"""Tests for container_utils."""
import collections
import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import attr

from tensorflow_federated.python.common_libs import named_containers


@dataclasses.dataclass
class TestDataclass:
  x: Any


@attr.s(auto_attribs=True)
class TestAttrClass:
  x: Any


class UserDefinedClassToOdictTest(parameterized.TestCase):

  def test_dataclass_converted_to_odict(self):
    a = TestDataclass(0)
    odict_a = named_containers.dataclass_to_odict(a)
    self.assertIsInstance(odict_a, collections.OrderedDict)
    self.assertLen(odict_a, 1)
    self.assertEqual(odict_a['x'], 0)

  @parameterized.named_parameters(
      ('list', lambda x: [x], list),
      ('odict', lambda a: collections.OrderedDict(x=a),
       collections.OrderedDict), ('dataclass', TestDataclass, TestDataclass),
      ('attrs_class', TestAttrClass, TestAttrClass))
  def test_dataclass_not_recursed_through(self, container_fn, expected_type):
    a = TestDataclass(container_fn(0))
    odict_a = named_containers.dataclass_to_odict(a)
    self.assertIsInstance(odict_a, collections.OrderedDict)
    self.assertLen(odict_a, 1)
    self.assertIsInstance(odict_a['x'], expected_type)

  def test_dataclass_conversion_raises_non_dataclass(self):
    odict = collections.OrderedDict(x=1)
    with self.assertRaises(TypeError):
      named_containers.dataclass_to_odict(odict)

  def test_attrs_class_converted_to_odict(self):
    a = TestAttrClass(0)
    odict_a = named_containers.attrs_class_to_odict(a)
    self.assertIsInstance(odict_a, collections.OrderedDict)
    self.assertLen(odict_a, 1)
    self.assertEqual(odict_a['x'], 0)

  @parameterized.named_parameters(
      ('list', lambda x: [x], list),
      ('odict', lambda a: collections.OrderedDict(x=a),
       collections.OrderedDict), ('dataclass', TestDataclass, TestDataclass),
      ('attrs_class', TestAttrClass, TestAttrClass))
  def test_attrs_class_not_recursed_through(self, container_fn, expected_type):
    a = TestAttrClass(container_fn(0))
    odict_a = named_containers.attrs_class_to_odict(a)
    self.assertIsInstance(odict_a, collections.OrderedDict)
    self.assertLen(odict_a, 1)
    self.assertIsInstance(odict_a['x'], expected_type)

  def test_attrs_class_conversion_raises_non_attrs_class(self):
    odict = collections.OrderedDict(x=1)
    with self.assertRaises(TypeError):
      named_containers.attrs_class_to_odict(odict)


if __name__ == '__main__':
  absltest.main()
