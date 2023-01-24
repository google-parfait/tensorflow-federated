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

import tensorflow as tf
import tensorflow_federated as tff


# b/263157965 : Extend this test to cover the C++ remote executor as well.
# pyformat: disable
_CONTEXTS = [
    ('native_sync_local_cpp',
     functools.partial(
         tff.backends.native.create_sync_local_cpp_execution_context,
         stream_structs=True)),
]
# pyformat: enable


class RemoteRuntimeStreamStructsTest(parameterized.TestCase):

  @tff.test.with_contexts(*_CONTEXTS)
  def test_large_struct_identity0(self):
    small_tensor_shape = (100_000_000, 1)
    large_struct = tff.structure.Struct(
        [(None, tf.zeros(shape=small_tensor_shape, dtype=tf.float32))] * 6
    )

    @tff.tf_computation(
        tff.StructType(
            [(None, tff.TensorType(shape=small_tensor_shape, dtype=tf.float32))]
            * 6
        )
    )
    def identity(s):
      with tf.compat.v1.control_dependencies(
          [tf.print(t) for t in tff.structure.flatten(s)]
      ):
        return tff.structure.map_structure(tf.identity, s)

    identity(large_struct)

  @tff.test.with_contexts(*_CONTEXTS)
  def test_large_struct_identity1(self):
    small_tensor_shape = (100_000, 1000)
    large_struct = tff.structure.Struct([
        ('a', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('b', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('c', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('d', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('e', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('f', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
    ])

    @tff.tf_computation(
        tff.StructType([
            (
                'a',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
            (
                'b',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
            (
                'c',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
            (
                'd',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
            (
                'e',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
            (
                'f',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
        ])
    )
    def identity(s):
      with tf.compat.v1.control_dependencies(
          [tf.print(t) for t in tff.structure.flatten(s)]
      ):
        return tff.structure.map_structure(tf.identity, s)

    identity(large_struct)

  @tff.test.with_contexts(*_CONTEXTS)
  def test_large_struct_identity2(self):
    small_tensor_shape = (100, 10)
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

    @tff.tf_computation(
        tff.StructType([
            (
                'a',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
            (
                'b',
                tff.StructType([
                    (
                        'b0',
                        tff.TensorType(
                            shape=small_tensor_shape, dtype=tf.float32
                        ),
                    ),
                    (
                        'b1',
                        tff.TensorType(
                            shape=small_tensor_shape, dtype=tf.float32
                        ),
                    ),
                ]),
            ),
            (
                'c',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
            (
                'd',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
            (
                'e',
                tff.TensorType(shape=small_tensor_shape, dtype=tf.float32),
            ),
        ])
    )
    def identity(s):
      with tf.compat.v1.control_dependencies(
          [tf.print(t) for t in tff.structure.flatten(s)]
      ):
        return tff.structure.map_structure(tf.identity, s)

    identity(small_struct)


if __name__ == '__main__':
  absltest.main()
