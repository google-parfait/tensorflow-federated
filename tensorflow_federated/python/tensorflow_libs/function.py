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
"""Methods for working with `tf.function` decorated functions."""


def is_tf_function(fn):
  """Determines whether `fn` is a method wrapped in `tf.function` decorator.

  Args:
    fn: The object to test.

  Returns:
    True iff `fn` is a function wrapped with a `tf.function` decorator.
  """
  # TODO(b/113112885): Add a cleaner way to check for
  # `tensorflow.python.eager.def_function.Function`, or whether a function is
  # wrapped in `tf.function`.
  return hasattr(fn, 'python_function')
