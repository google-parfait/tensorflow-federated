# Copyright 2020, The TensorFlow Federated Authors.
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
"""Library of helper functions for working with TensorFlow `tf.Variable`."""

import contextlib

import tensorflow as tf


@contextlib.contextmanager
def record_variable_creation_scope():
  """Creates a single use contextmanager for capture variable creation calls."""
  variable_list = []

  def logging_variable_creator(next_creator, **kwargs):
    variable = next_creator(**kwargs)
    variable_list.append(variable)
    return variable

  with contextlib.ExitStack() as stack:
    stack.enter_context(tf.variable_creator_scope(logging_variable_creator))
    yield variable_list
