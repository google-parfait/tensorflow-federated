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
"""General purpose test utilities for TFF."""

from absl import logging
import tensorflow.compat.v2 as tf

from tensorflow_federated.python.core.api import computations

tf.compat.v1.enable_v2_behavior()  # Required to create a local executor.


def tf1_and_tf2(fn):
  """A decorator for creating test parameterized by TF computation decorators.

  Args:
    fn: A test function to be decorated. It must accept two arguments: self (a
      `TestCase`), and tf_computation (either a `tff.tf_computation` or
      `tff.tf2_computation`).

  Returns:
    A decorated function, which executes `fn` using both decorators.
  """

  def wrapped_fn(self):
    logging.info('Testing under tff.tf2_computation')
    fn(self, computations.tf2_computation)
    logging.info('Testing under tff.tf_computation')
    fn(self, computations.tf_computation)

  return wrapped_fn


def tf1(fn):
  """A decorator for testing the `tff.tf_computation` decorator."""

  def wrapped_fn(self):
    fn(self, computations.tf_computation)

  return wrapped_fn


def tf2(fn):
  """A decorator for testing the `tff.tf2_computation` decorator."""

  def wrapped_fn(self):
    fn(self, computations.tf2_computation)

  return wrapped_fn
