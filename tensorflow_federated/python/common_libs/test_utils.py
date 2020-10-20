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
"""General purpose test utils for TFF."""

import functools

import tensorflow as tf


def graph_mode_test(test_fn):
  """Decorator for a test to be executed in graph mode.

  This decorator is used to write graph-mode tests when eager execution is
  enabled.

  This introduces a default `tf.Graph`, which tests annotated with
  `@graph_mode_test` may use or ignore by creating their own Graphs.

  Args:
    test_fn: A test function to be decorated.

  Returns:
    The decorated test_fn.
  """

  @functools.wraps(test_fn)
  def wrapped_test_fn(*args, **kwargs):
    with tf.Graph().as_default():
      test_fn(*args, **kwargs)

  return wrapped_test_fn


# TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
def skip_test_for_gpu(test_fn):
  """Decorator for a test to be skipped in GPU tests.

  Args:
    test_fn: A test function to be decorated.

  Returns:
    The decorated test_fn.
  """

  @functools.wraps(test_fn)
  def wrapped_test_fn(self, *args, **kwargs):
    gpu_devices = tf.config.list_logical_devices('GPU')
    if gpu_devices:
      self.skipTest('skip GPU test')
    test_fn(self, *args, **kwargs)

  return wrapped_test_fn


def create_logical_multi_gpus(memory_limit=128):
  """Create a (virtual )multi-GPU environment for testing."""
  # Multiple logical GPU devices will be created for TFF simulation. Only call
  # this function once in a module as logical deviceds have to be created before
  # listed in each indivisual test. Typically in `setUp` as
  # `list_physical_devices` cannot be directly called in `main`.
  gpu_devices = tf.config.list_physical_devices('GPU')
  if not gpu_devices:
    raise ValueError('Physical GPU is not detected.')
  if len(gpu_devices) == 1:
    tf.config.set_logical_device_configuration(gpu_devices[0], [
        tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit),
        tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)
    ])


# TODO(b/160896627): Kokoro GPU tests provide multi-GPU environment by default,
# we use this decorator to skip dataset.reduce in multi-GPU environment.
def skip_test_for_multi_gpu(test_fn):
  """Decorator for a test to be skipped in multi GPU environment.

  Args:
    test_fn: A test function to be decorated.

  Returns:
    The decorated test_fn.
  """

  @functools.wraps(test_fn)
  def wrapped_test_fn(self, *args, **kwargs):
    gpu_devices = tf.config.list_logical_devices('GPU')
    if len(gpu_devices) > 1:
      self.skipTest('skip GPU test')
    test_fn(self, *args, **kwargs)

  return wrapped_test_fn
