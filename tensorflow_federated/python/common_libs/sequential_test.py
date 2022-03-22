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

import re

from absl import logging
import tensorflow as tf

from tensorflow_federated.python.common_libs import sequential

UNIT_COUNT = 10  # Number of computational units to use for tests
UNIT_BYTES = 100_000_000  # 100 MB


def get_high_watermark_bytes():
  with open('/proc/self/status') as f:
    lines = f.readlines()
  pattern = re.compile(r'VmHWM\:\s*(\d+)\s*kB\n')
  for line in lines:
    match = pattern.fullmatch(line)
    if match:
      hwm_kb = int(match.group(1))
      return hwm_kb * 1024
  raise KeyError('Could not find "VmHWM" record in /proc/self/status.')


def reset_high_watermark():
  with open('/proc/self/clear_refs', 'w') as f:
    f.write('5')


def memory_intense_fn(x):
  """Function which should take about UNIT_BYTES (100 MB) of memory to execute.

  The function creates a large random tensor under the hood which is responsible
  for the bulk of the memory consumption.

  Args:
    x: Input scalar tensor

  Returns:
    Scalar tensor UNIT_BYTES * x + some random noise.
  """
  matrix = tf.random.uniform([UNIT_BYTES // 4])  # float32 is 4B
  matrix = matrix + x
  return tf.reduce_sum(matrix)


class SequentialWrapperTest(tf.test.TestCase):

  def test_high_watermark_reset_call_resets_the_high_waterark(self):

    @tf.function()
    def memory_intense_processing(structure):
      return tf.nest.map_structure(memory_intense_fn, structure)

    # Use temporarily a lot of memory
    x = tuple(range(UNIT_COUNT))
    memory_intense_processing(x)
    hwm_before = get_high_watermark_bytes()
    reset_high_watermark()

    # Measure memory after the memory intensive computation is over
    hwm_after = get_high_watermark_bytes()

    self.assertGreater(hwm_before, hwm_after)

  def test_tf_nest_map_structure_uses_high_memory_due_to_parallelism(self):

    @tf.function()
    def tf_nest_map_stucture(x):
      return tf.nest.map_structure(memory_intense_fn, x)

    # Trace the function
    x = tuple(range(UNIT_COUNT))
    tf_nest_map_stucture(x)

    # Measured execution
    reset_high_watermark()
    hwm_before = get_high_watermark_bytes()
    tf_nest_map_stucture(x)
    hwm_after = get_high_watermark_bytes()

    hwm_diff = hwm_after - hwm_before
    logging.info('parallel execution bytes: %d (initial: %d)', hwm_diff,
                 hwm_before)
    self.assertGreater(hwm_diff, 4 * UNIT_BYTES)  # hwm_diff is about 900 MB

  def test_sequential_wrapper_reduces_memory_by_restricting_parallelism(self):
    reset_high_watermark()
    hwm_before = get_high_watermark_bytes()

    @tf.function()
    def sequential_nest_map_stucture(structure):
      return tf.nest.map_structure(
          sequential.SequentialWrapper(memory_intense_fn), structure)

    # Trace the function
    x = tuple(range(UNIT_COUNT))
    sequential_nest_map_stucture(x)

    # Measured execution
    reset_high_watermark()
    hwm_before = get_high_watermark_bytes()
    sequential_nest_map_stucture(x)
    hwm_after = get_high_watermark_bytes()

    hwm_diff = hwm_after - hwm_before
    logging.info('sequential execution bytes: %d (initial: %d)', hwm_diff,
                 hwm_before)
    self.assertLess(hwm_diff, 2 * UNIT_BYTES)  # hwm_diff is usually 0-180 MB

  def test_sequential_wrapper_doesnt_affect_functionality(self):
    x = (0, (1, 2))

    @sequential.SequentialWrapper
    def add_one(x):
      return x + 1

    @tf.function
    def add_1_to_all(structure):
      return tf.nest.map_structure(add_one, structure)

    y = add_1_to_all(x)
    self.assertAllEqual(y, (1, (2, 3)))


if __name__ == '__main__':
  tf.test.main()
