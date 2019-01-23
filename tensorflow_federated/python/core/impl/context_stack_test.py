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

from absl.testing import absltest

from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import reference_executor


class TestContext(context_base.Context):

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    return self._name

  def ingest(self, val, type_spec):
    raise NotImplementedError

  def invoke(self, comp, arg):
    return NotImplementedError


class ContextStackTest(absltest.TestCase):

  def test_basic_functionality(self):
    ctx_stack = context_stack_impl.context_stack
    self.assertIsInstance(ctx_stack, context_stack_impl.ContextStackImpl)
    self.assertIsInstance(ctx_stack.current,
                          reference_executor.ReferenceExecutor)

    with ctx_stack.install(TestContext('foo')):
      self.assertIsInstance(ctx_stack.current, TestContext)
      self.assertEqual(ctx_stack.current.name, 'foo')

      with ctx_stack.install(TestContext('bar')):
        self.assertIsInstance(ctx_stack.current, TestContext)
        self.assertEqual(ctx_stack.current.name, 'bar')

      self.assertEqual(ctx_stack.current.name, 'foo')

    self.assertIsInstance(ctx_stack.current,
                          reference_executor.ReferenceExecutor)


if __name__ == '__main__':
  absltest.main()
