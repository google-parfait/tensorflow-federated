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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


_CONTEXTS = [
    (
        'native_sync_local_cpp',
        functools.partial(
            tff.backends.native.create_sync_local_cpp_execution_context,
            stream_structs=True,
        ),
    ),
]


def _make_federated(computation: tff.Computation) -> tff.Computation:
  """Construct a federate computation that maps comptuation to CLIENTS."""

  @tff.federated_computation(
      tff.FederatedType(computation.type_signature.parameter, tff.CLIENTS),
  )
  def compute(a):
    return tff.federated_map(computation, a)

  return compute


class RemoteRuntimeStreamStructsTest(parameterized.TestCase):

  @tff.test.with_contexts(*_CONTEXTS)
  def test_large_unnamed_struct_identity(self):
    # Expect ~400 MB per small tensor (100_000_000 * 4 bytes)
    small_tensor_shape = (100_000_000, 1)
    # Expect 2.4 GB in total for the entire structure.
    large_struct = tff.structure.Struct(
        [(None, tf.zeros(shape=small_tensor_shape, dtype=tf.float32))] * 6
    )

    @tff.tensorflow.computation(
        tff.StructType(
            [(None, tff.TensorType(np.float32, small_tensor_shape))] * 6
        )
    )
    def identity(s):
      return tff.structure.map_structure(tf.identity, s)

    with self.subTest('local'):
      identity(large_struct)

  @tff.test.with_contexts(*_CONTEXTS)
  def test_large_named_struct_identity(self):
    # Expect ~400 MB per small tensor (100_000_000 * 4 bytes)
    small_tensor_shape = (100_000, 1000)
    # Expect 2.4 GB in total for the entire structure.
    large_struct = tff.structure.Struct([
        ('a', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('b', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('c', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('d', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('e', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('f', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
    ])

    @tff.tensorflow.computation(
        tff.StructType([
            (
                'a',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'b',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'c',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'd',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'e',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'f',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
        ])
    )
    def identity(s):
      return tff.structure.map_structure(tf.identity, s)

    with self.subTest('local'):
      identity(large_struct)

  @tff.test.with_contexts(*_CONTEXTS)
  def test_small_struct_identity(self):
    # Expect ~4KB per small tensor.
    small_tensor_shape = (100, 10)
    # Expect ~24KB for the entire structure.
    small_struct = tff.structure.Struct([
        ('a', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        (
            'b',
            tff.structure.Struct([
                (
                    'b0',
                    tf.zeros(shape=small_tensor_shape, dtype=tf.float32),
                ),
                (
                    'b1',
                    tf.zeros(shape=small_tensor_shape, dtype=tf.float32),
                ),
            ]),
        ),
        ('c', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('d', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('e', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
    ])

    @tff.tensorflow.computation(
        tff.StructType([
            (
                'a',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'b',
                tff.StructType([
                    (
                        'b0',
                        tff.TensorType(np.float32, small_tensor_shape),
                    ),
                    (
                        'b1',
                        tff.TensorType(np.float32, small_tensor_shape),
                    ),
                ]),
            ),
            (
                'c',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'd',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'e',
                tff.TensorType(np.float32, small_tensor_shape),
            ),
        ])
    )
    def identity(s):
      return tff.structure.map_structure(tf.identity, s)

    with self.subTest('local'):
      identity(small_struct)


if __name__ == '__main__':
  absltest.main()
