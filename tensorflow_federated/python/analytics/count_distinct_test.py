# Copyright 2023, The TensorFlow Federated Authors.
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
"""Tests for count_distinct.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics import count_distinct
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types


def hll_sketch_python(values):
  """Computes a HyperLogLog sketch from a list of hash values.

  This function is intended to be used to verify the behavior of
  hyperloglog._client_hyperloglog, which is a pure tensorflow implementation of
  the same algorithm with some bit-manipulation magic rather than string
  processing of binary numbers as we do here.

  This implementation is supposed to be 'obviously correct' based on a direct
  python translation of the steps outlined in
  https://en.wikipedia.org/wiki/HyperLogLog.

  Args:
    values: a list of int64 hash values.

  Returns:
    The HyperLogLog sketch, an array of size HLL_SKETCH_SIZE.
  """
  ans = np.zeros(count_distinct.HLL_SKETCH_SIZE, dtype=np.int32)
  for val in values:
    binary = bin(val)[2:].zfill(count_distinct.HLL_SKETCH_SIZE)
    # not adding 1 here because array is 0-indexed
    j = int(binary[: count_distinct.HLL_BIT_INDEX_HEAD], 2)
    w = binary[count_distinct.HLL_BIT_INDEX_HEAD :]
    # position of left-most 1, starting at 1.  Returns 0 if 1 is not present
    rho_w = w.find('1') + 1
    ans[j] = max(ans[j], rho_w)
  return ans


class CountDistinctComputationTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_test_cpp_execution_context()

  def test_constants(self):
    self.assertIn(count_distinct.HLL_SKETCH_SIZE, [16, 32, 64])
    alpha = {16: 0.673, 32: 0.697, 64: 0.709}[count_distinct.HLL_SKETCH_SIZE]
    self.assertEqual(count_distinct.HLL_ALPHA, alpha)
    head = {16: 4, 32: 5, 64: 6}[count_distinct.HLL_SKETCH_SIZE]
    self.assertEqual(count_distinct.HLL_BIT_INDEX_HEAD, head)

  def test_type_properties(self):

    client_hyperloglog = count_distinct.build_client_hyperloglog_computation()

    expected_type_signature = computation_types.FunctionType(
        computation_types.SequenceType(computation_types.TensorType(tf.int64)),
        computation_types.TensorType(
            tf.int64, [count_distinct.HLL_SKETCH_SIZE]
        ),
    )
    self.assertTrue(
        client_hyperloglog.type_signature.is_identical_to(
            expected_type_signature
        )
    )


class CountDistinctExecutionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_test_cpp_execution_context()

  @parameterized.named_parameters(('default', False), ('secure', True))
  def test_runs_end_to_end_without_error(self, secagg=False):
    data = [['a', 'b', 'c', 'a', 'c'], ['c', 'd', 'd', 'c'], ['a', 'd']]
    hll = count_distinct.create_federated_hyperloglog_computation(
        use_secagg=secagg
    )
    count = hll(data)
    self.assertShapeEqual(count, np.empty(()))

  def test_against_python_implementation(self):
    prng = np.random.RandomState(12345)
    hashed_data = prng.randint(0, 2**count_distinct.HLL_SKETCH_SIZE, size=100)
    fake_hashed_data = tf.data.Dataset.from_tensor_slices(hashed_data)
    client_hyperloglog = count_distinct.build_client_hyperloglog_computation()

    sketch = client_hyperloglog(fake_hashed_data)
    true_sketch = hll_sketch_python(hashed_data)
    self.assertEqual(sketch.shape, (count_distinct.HLL_SKETCH_SIZE,))
    self.assertAllEqual(sketch, true_sketch)

  def test_federated_secure_max(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    input1 = [3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,0,0,0,0,0]
    input2 = [2,7,1,8,2,8,1,8,2,8,4,5,9,0,4,5,2,3,5,3,6,0,2,8,7,4,7,0,0,0,0,0]
    input3 = [1,6,1,8,0,3,3,9,8,8,7,4,9,8,9,4,8,4,8,2,0,4,5,8,6,8,3,0,0,0,0,0]
    expect = [3,7,4,8,5,9,3,9,8,8,7,8,9,8,9,5,8,4,8,4,6,4,6,8,7,8,8,0,0,0,0,0]
    # pyformat: enable
    # pylint: enable=bad-whitespace

    federated_max = count_distinct.build_federated_secure_max_computation()

    answer = federated_max([input1, input2, input3])
    self.assertAllEqual(answer, expect)


if __name__ == '__main__':
  tf.test.main()
