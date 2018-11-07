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
"""Tests for the ContextStack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from tensorflow_federated.python.core.impl.context_base import Context
from tensorflow_federated.python.core.impl.context_stack import context_stack
from tensorflow_federated.python.core.impl.default_context import DefaultContext


class TestContext(Context):

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    return self._name

  def invoke(self, comp, arg):
    return NotImplementedError


class ContextStackTest(unittest.TestCase):

  def test_basic_functionality(self):

    self.assertIsInstance(context_stack.current, DefaultContext)

    with context_stack.install(TestContext('foo')):
      self.assertIsInstance(context_stack.current, TestContext)
      self.assertEqual(context_stack.current.name, 'foo')

      with context_stack.install(TestContext('bar')):
        self.assertIsInstance(context_stack.current, TestContext)
        self.assertEqual(context_stack.current.name, 'bar')

      self.assertEqual(context_stack.current.name, 'foo')

    self.assertIsInstance(context_stack.current, DefaultContext)


if __name__ == '__main__':
  unittest.main()
