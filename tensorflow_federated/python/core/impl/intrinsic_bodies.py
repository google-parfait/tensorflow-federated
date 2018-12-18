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
"""Bodies of intrinsics to be added as replacements by the compiler pipleine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import intrinsic_factory
from tensorflow_federated.python.core.impl import intrinsic_utils


def get_intrinsic_bodies(context_stack):
  """Returns a dictionary of intrinsic bodies.

  Args:
    context_stack: The context stack to use.
  """
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  intrinsics = intrinsic_factory.IntrinsicFactory(context_stack)

  def federated_sum(x):
    zero = intrinsic_utils.zero_for(x.type_signature.member, context_stack)
    plus = intrinsic_utils.plus_for(x.type_signature.member, context_stack)
    return intrinsics.federated_reduce(x, zero, plus)

  return collections.OrderedDict([(intrinsic_defs.FEDERATED_SUM.uri,
                                   federated_sum)])
