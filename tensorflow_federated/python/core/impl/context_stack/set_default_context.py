# Lint as: python3
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
"""A utility to change the context stack."""

from tensorflow_federated.python.core.impl.context_stack import context_stack_impl


def set_default_context(ctx=None):
  """Places a context at the top of the context stack.

  Args:
    ctx: Either an instance of `context_base.Context`, or `None`, with the
      latter resulting in the default reference executor getting installed at
      the bottom of the stack (as is the default).
  """
  context_stack_impl.context_stack.set_default_context(ctx)
