# Copyright 2020, The TensorFlow Federated Authors.
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
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff


def get_all_execution_contexts():
  return [
      ('native_local', tff.backends.native.create_local_execution_context()),
      ('native_sizing', tff.backends.native.create_sizing_execution_context()),
      ('native_debug',
       tff.backends.native.create_thread_debugging_execution_context()),
  ]


def with_contexts(*args):
  """A decorator for creating tests parameterized by context.

  Note: To use this decorator your test is required to inherit from
  `parameterized.TestCase`.

  The decorator can be called without arguments:

  ```
  @with_contexts
  def foo(self):
    ...
  ```

  or with arguments:

  ```
  @with_contexts(
      ('label', executor),
      ...
  )
  def foo(self):
    ...
  ```

  If the decorator is specified without arguments or is called with no
  arguments, the default contexts used are those returned by
  `get_all_execution_contexts`.

  If the decorator is called with arguments the arguments must be in a form that
  is accpeted by `parameterized.named_parameters`.

  Args:
    *args: Either a test function to be decorated or named executors for the
      decorated method, either a single iterable, or a list of tuples or dicts.

  Returns:
     A test generator to be handled by `parameterized.TestGeneratorMetaclass`.
  """

  def decorator(fn, *named_contexts):
    if not named_contexts:
      named_contexts = get_all_execution_contexts()

    @parameterized.named_parameters(*named_contexts)
    def wrapped_fn(self, context):
      context_stack = tff.framework.get_context_stack()
      with context_stack.install(context):
        fn(self)

    return wrapped_fn

  if len(args) == 1 and callable(args[0]):
    return decorator(args[0])
  else:
    return lambda fn: decorator(fn, *args)


class ExecutionContextsTest(parameterized.TestCase):

  @with_contexts
  def test_federated_value(self):

    @tff.federated_computation
    def foo(x):
      return tff.federated_value(x, tff.SERVER)

    result = foo(10)
    self.assertIsNotNone(result)

  @with_contexts
  def test_federated_zip(self):

    @tff.federated_computation([tff.FederatedType(tf.int32, tff.CLIENTS)] * 2)
    def foo(x):
      return tff.federated_zip(x)

    result = foo([[1, 2], [3, 4]])
    self.assertIsNotNone(result)

  @with_contexts
  def test_federated_zip_with_twenty_elements(self):
    # This test will fail if execution scales factorially with number of
    # elements zipped.
    num_element = 20
    num_clients = 2

    @tff.federated_computation([tff.FederatedType(tf.int32, tff.CLIENTS)] *
                               num_element)
    def foo(x):
      return tff.federated_zip(x)

    value = [list(range(num_clients))] * num_element
    result = foo(value)
    self.assertIsNotNone(result)

  @with_contexts
  def test_identity(self):

    @tff.federated_computation
    def foo(x):
      return x

    result = foo(10)
    self.assertIsNotNone(result)


if __name__ == '__main__':
  tff.test.set_no_default_context()
  absltest.main()
