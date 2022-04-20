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
"""Tests for `async_utils` library."""

import asyncio

from absl.testing import absltest

from tensorflow_federated.python.common_libs import async_utils


async def _sleep_then_append(entries, value):
  await asyncio.sleep(0.01)
  entries.append(value)


class OrderedTasksTest(absltest.TestCase):

  def test_runs_awaitable(self):
    nums = []

    async def run():
      async with async_utils.ordered_tasks() as tasks:
        tasks.add(_sleep_then_append(nums, 0))

    asyncio.run(run())
    self.assertEqual(nums, [0])

  def test_runs_awaitable_in_separate_task_while_in_context(self):

    async def run():
      event = asyncio.Event()

      async def awaitable():
        await asyncio.sleep(0.01)
        event.set()

      async with async_utils.ordered_tasks() as tasks:
        tasks.add(awaitable())
        await asyncio.wait_for(event.wait(), timeout=10)

    asyncio.run(run())

  def test_runs_despite_exception_and_logs(self):
    nums = []

    async def _sleep_then_append_then_throw(entries, value):
      await asyncio.sleep(0.01)
      entries.append(value)
      raise ValueError('async throwing')

    def _append_then_throw():
      nums.append(1)
      raise ValueError('sync throwing')

    async def run():
      async with async_utils.ordered_tasks() as tasks:
        tasks.add(_sleep_then_append_then_throw(nums, 0))
        tasks.add_callable(_append_then_throw)
        tasks.add(_sleep_then_append_then_throw(nums, 2))

    with self.assertLogs() as logs:
      asyncio.run(run())
    self.assertEqual(nums, [0, 1, 2])
    self.assertLen(logs.records, 3)

  def test_runs_awaitables(self):
    nums = []

    async def run():
      async with async_utils.ordered_tasks() as tasks:
        tasks.add_all(*(_sleep_then_append(nums, n) for n in range(3)))

    asyncio.run(run())
    self.assertSameElements(nums, list(range(3)))

  def test_runs_sync_after_awaitable(self):
    nums = []

    async def run():
      async with async_utils.ordered_tasks() as tasks:
        tasks.add(_sleep_then_append(nums, 0))
        tasks.add_callable(lambda: nums.append(1))

    asyncio.run(run())
    self.assertEqual(nums, [0, 1])


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
