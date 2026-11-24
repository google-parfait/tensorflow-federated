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

from collections.abc import Iterator
import contextlib
import os
import socket
import subprocess
import time

from absl.testing import absltest
import federated_language
import numpy as np
import portpicker
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.core.backends.native import cpp_execution_contexts
from tensorflow_federated.python.core.impl.executors import executor_bindings

_WORKER_HOST = 'localhost'
_WORKER_STARTUP_TIMEOUT_SECONDS = 30.0
_WORKER_POLL_INTERVAL_SECONDS = 0.05
_WORKER_CONNECT_TIMEOUT_SECONDS = 0.1
_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def _worker_binary_path() -> str:
  return os.path.join(
      os.path.dirname(__file__), '..', '..', 'cc', 'simulation', 'worker_binary'
  )


def _wait_for_port_ready(
    process: subprocess.Popen[bytes], port: int, timeout_seconds: float
) -> None:
  deadline = time.monotonic() + timeout_seconds
  while time.monotonic() < deadline:
    if process.poll() is not None:
      raise RuntimeError(
          f'Worker exited prematurely with code {process.returncode}.'
      )
    try:
      with socket.create_connection(
          (_WORKER_HOST, port), timeout=_WORKER_CONNECT_TIMEOUT_SECONDS
      ):
        return
    except OSError:
      time.sleep(_WORKER_POLL_INTERVAL_SECONDS)
  raise TimeoutError(
      f'Timed out after {timeout_seconds}s waiting for worker on port {port}.'
  )


@contextlib.contextmanager
def _remote_streaming_cpp_context() -> (
    Iterator[federated_language.framework.SyncExecutionContext]
):
  port = portpicker.pick_unused_port()
  process = subprocess.Popen([_worker_binary_path(), f'--port={port}'])
  try:
    _wait_for_port_ready(process, port, _WORKER_STARTUP_TIMEOUT_SECONDS)
    channel = executor_bindings.create_insecure_grpc_channel(
        f'{_WORKER_HOST}:{port}'
    )
    context = cpp_execution_contexts.create_sync_remote_cpp_execution_context(
        channels=[channel], stream_structs=True
    )
    with federated_language.framework.get_context_stack().install(context):
      yield context
  finally:
    process.terminate()
    try:
      process.wait(timeout=_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
      process.kill()
      process.wait()


def _make_federated(
    computation: federated_language.Computation,
) -> federated_language.Computation:
  """Construct a federate computation that maps comptuation to CLIENTS."""

  @federated_language.federated_computation(
      federated_language.FederatedType(
          computation.type_signature.parameter, federated_language.CLIENTS
      ),
  )
  def compute(a):
    return federated_language.federated_map(computation, a)

  return compute


class RemoteRuntimeStreamStructsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(_remote_streaming_cpp_context())

  def test_large_unnamed_struct_identity(self):
    self.skipTest('b/410811127')

    # Expect ~400 MB per small tensor (100_000_000 * 4 bytes)
    small_tensor_shape = (100_000_000, 1)
    # Expect 2.4 GB in total for the entire structure.
    large_struct = [tf.zeros(shape=small_tensor_shape, dtype=tf.float32)] * 6

    @tff.tensorflow.computation(
        federated_language.StructType(
            [(
                None,
                federated_language.TensorType(np.float32, small_tensor_shape),
            )]
            * 6
        )
    )
    def identity(s):
      return tf.identity(s)

    with self.subTest('local'):
      identity(large_struct)

    with self.subTest('federated'):
      federated_identity = _make_federated(identity)
      federated_identity([large_struct])

  def test_large_named_struct_identity(self):
    self.skipTest('b/410811127')

    # Expect ~400 MB per small tensor (100_000_000 * 4 bytes)
    small_tensor_shape = (100_000, 1000)
    # Expect 2.4 GB in total for the entire structure.
    large_struct = [
        ('a', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('b', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('c', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('d', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('e', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('f', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
    ]

    @tff.tensorflow.computation(
        federated_language.StructType([
            (
                'a',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'b',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'c',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'd',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'e',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'f',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
        ])
    )
    def identity(s):
      return tf.identity(s)

    with self.subTest('local'):
      identity(large_struct)

    with self.subTest('federated'):
      federated_identity = _make_federated(identity)
      federated_identity([large_struct])

  def test_small_struct_identity(self):
    self.skipTest('b/410811127')

    # Expect ~4KB per small tensor.
    small_tensor_shape = (100, 10)
    # Expect ~24KB for the entire structure.
    small_struct = [
        ('a', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        (
            'b',
            [
                (
                    'b0',
                    tf.zeros(shape=small_tensor_shape, dtype=tf.float32),
                ),
                (
                    'b1',
                    tf.zeros(shape=small_tensor_shape, dtype=tf.float32),
                ),
            ],
        ),
        ('c', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('d', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('e', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
    ]

    @tff.tensorflow.computation(
        federated_language.StructType([
            (
                'a',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'b',
                federated_language.StructType([
                    (
                        'b0',
                        federated_language.TensorType(
                            np.float32, small_tensor_shape
                        ),
                    ),
                    (
                        'b1',
                        federated_language.TensorType(
                            np.float32, small_tensor_shape
                        ),
                    ),
                ]),
            ),
            (
                'c',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'd',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
            (
                'e',
                federated_language.TensorType(np.float32, small_tensor_shape),
            ),
        ])
    )
    def identity(s):
      return tf.identity(s)

    with self.subTest('local'):
      identity(small_struct)

    with self.subTest('federated'):
      federated_identity = _make_federated(identity)
      federated_identity([small_struct])

  def test_deeply_nested_struct_identity(self):
    c_expected = np.arange(10, dtype=np.int32)
    d_expected = np.ones(shape=(2, 3), dtype=np.float32) * 5.0
    deeply_nested_struct = {
        'a': {
            'b': {
                'c': tf.constant(c_expected),
            },
            'd': tf.constant(d_expected),
        },
    }

    @tff.tensorflow.computation(
        federated_language.to_type({
            'a': {
                'b': {
                    'c': federated_language.TensorType(np.int32, (10,)),
                },
                'd': federated_language.TensorType(np.float32, (2, 3)),
            },
        })
    )
    def identity(s):
      return federated_language.common_libs.structure.map_structure(
          tf.identity, s
      )

    with self.subTest('local'):
      local_result = identity(deeply_nested_struct)
      np.testing.assert_array_equal(local_result['a']['b']['c'], c_expected)
      np.testing.assert_array_equal(local_result['a']['d'], d_expected)

    with self.subTest('federated'):
      federated_identity = _make_federated(identity)
      result = federated_identity([deeply_nested_struct])
      self.assertLen(result, 1)
      np.testing.assert_array_equal(result[0]['a']['b']['c'], c_expected)
      np.testing.assert_array_equal(result[0]['a']['d'], d_expected)


if __name__ == '__main__':
  absltest.main()
