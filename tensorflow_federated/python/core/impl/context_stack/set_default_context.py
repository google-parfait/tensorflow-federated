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
from tensorflow_federated.python.core.impl.context_stack import runtime_error_context


def set_default_context(ctx):
  """Places `ctx` at the bottom of the stack.

  Args:
    ctx: An instance of `context_base.Context`.
  """
  context_stack_impl.context_stack.set_default_context(ctx)


def set_no_default_context():
  """Places a `RuntimeErrorContext` at the bottom of the stack."""
  set_default_context(runtime_error_context.RuntimeErrorContext())
