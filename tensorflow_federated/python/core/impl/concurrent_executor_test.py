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

import asyncio
import collections
import time

from absl.testing import absltest
import tensorflow.compat.v2 as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import concurrent_executor
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import executor_base


class ConcurrentExecutorTest(absltest.TestCase):

  def test_nondeterminism_with_fake_executor_that_synchronously_sleeps(self):

    class FakeExecutor(executor_base.Executor):

      def __init__(self):
        self._values = []

      @property
      def output(self):
        return ''.join([str(x) for x in self._values])

      async def create_value(self, value, type_spec=None):
        del type_spec
        for _ in range(3):
          time.sleep(1)
          self._values.append(value)
        return value

      async def create_call(self, comp, arg=None):
        raise NotImplementedError

      async def create_tuple(self, elements):
        raise NotImplementedError

      async def create_selection(self, source, index=None, name=None):
        raise NotImplementedError

    def make_output():
      test_ex = FakeExecutor()
      executors = [
          concurrent_executor.ConcurrentExecutor(test_ex) for _ in range(10)
      ]
      loop = asyncio.get_event_loop()
      vals = [ex.create_value(idx) for idx, ex in enumerate(executors)]
      results = loop.run_until_complete(asyncio.gather(*vals))
      self.assertCountEqual(list(results), list(range(10)))
      del executors
      return test_ex.output

    o1 = make_output()
    for _ in range(1000):
      o2 = make_output()
      if o2 != o1:
        break
    self.assertNotEqual(o1, o2)

  def test_with_eager_executor(self):

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return tf.add(x, 1)

    ex = concurrent_executor.ConcurrentExecutor(eager_executor.EagerExecutor())

    async def compute():
      return await ex.create_selection(
          await ex.create_tuple(
              collections.OrderedDict([
                  ('a', await
                   ex.create_call(await ex.create_value(add_one), await
                                  ex.create_value(10, tf.int32)))
              ])),
          name='a')

    result = asyncio.get_event_loop().run_until_complete(compute())
    self.assertIsInstance(result, eager_executor.EagerValue)
    self.assertEqual(result.internal_representation.numpy(), 11)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
