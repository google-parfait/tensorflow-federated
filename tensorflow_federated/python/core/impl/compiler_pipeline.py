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
"""A pipeline that reduces computations into an executable form."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import transformations
from tensorflow_federated.python.core.impl import value_transformations


class CompilerPipeline(object):
  """The compiler pipeline.

  This pipeline will eventually be made configurable, and driven largely by what
  the targeted backend can support. The set of conversions that are currently
  being performed is fixed, and includes the following:

  1. Replacing occurrences of a subset of intrinsics with their definitions in
     terms of other intrinsics, as defined in `intrinsic_bodies.py`.
  """

  def __init__(self, context_stack):
    """Constructs this pipeline with the given dictionary of intrinsic bodies.

    Args:
      context_stack: The context stack to use.
    """
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    self._context_stack = context_stack

  def compile(self, computation_to_compile):
    """Compiles `computation_to_compile`.

    Args:
      computation_to_compile: An instance of `computation_base.Computation` to
        compile.

    Returns:
      An instance of `computation_base.Computation` that repeesents the result.
    """
    py_typecheck.check_type(computation_to_compile,
                            computation_base.Computation)
    computation_proto = computation_impl.ComputationImpl.get_proto(
        computation_to_compile)

    # TODO(b/113123410): Add a compiler options argument that characterizes the
    # desired form of the output. To be driven by what the specific backend the
    # pipeline is targeting is able to understand. Pending a more fleshed out
    # design of the backend API.

    py_typecheck.check_type(computation_proto, pb.Computation)
    comp = computation_building_blocks.ComputationBuildingBlock.from_proto(
        computation_proto)

    # Replace intrinsics with their bodies, for now manually in a fixed order.
    # TODO(b/113123410): Replace this with a more automated implementation that
    # does not rely on manual maintenance.
    comp, _ = value_transformations.replace_all_intrinsics_with_bodies(
        comp, self._context_stack)

    # Replaces called lambdas with LET constructs with a single local symbol.
    comp, _ = transformations.replace_called_lambda_with_block(comp)
    # TODO(b/113123410): Add more transformations to simplify and optimize the
    # structure, e.g., such as:
    # * removing unnecessary lambdas,
    # * flatteting the structure,
    # * merging TensorFlow blocks where appropriate,
    # * ...and so on.

    return computation_impl.ComputationImpl(comp.proto, self._context_stack)
