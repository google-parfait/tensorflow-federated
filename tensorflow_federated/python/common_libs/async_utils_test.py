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
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

from absl.testing import absltest

from tensorflow_federated.python.common_libs import async_utils


class SharedAwaitableTest(absltest.TestCase):

  async def async_five(self):
    return 5

  def test_await_once_returns_result(self):
    async def x():
      self.assertEqual(await async_utils.SharedAwaitable(self.async_five()), 5)

    asyncio.run(x())

  def test_await_several_times_returns_result(self):
    async def x():
      async_result = async_utils.SharedAwaitable(self.async_five())
      self.assertEqual(await async_result, 5)
      self.assertEqual(await async_result, 5)
      self.assertEqual(await async_result, 5)
      self.assertEqual(await async_result, 5)

    asyncio.run(x())

  async def async_raise_value_error(self):
    raise ValueError('error')

  def test_reraises_exception_on_await(self):
    async def x():
      async_raiser = async_utils.SharedAwaitable(self.async_raise_value_error())
      with self.assertRaises(ValueError):
        await async_raiser

    asyncio.run(x())


if __name__ == '__main__':
  absltest.main()
