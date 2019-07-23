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
"""Intrinsic reductions as AST transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import value_transformations


def replace_intrinsics_with_bodies(comp):
  """Reduces intrinsics to their bodies as defined in `intrinsic_bodies.py`.

  This function operates on the AST level; meaning, it takes in a
  `computation_building_blocks.ComputationBuildingBlock` as an argument and
  returns one as well. `replace_intrinsics_with_bodies` is intended to be the
  standard reduction function, which will reduce all currently implemented
  intrinsics to their bodies.

  Notice that the success of this function depends on the contract of
  `intrinsic_bodies.get_intrinsic_bodies`, that the dict returned by that
  function is ordered from more complex intrinsic to less complex intrinsics.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock` in
      which we wish to replace all intrinsics with their bodies.

  Returns:
    An instance of `computation_building_blocks.ComputationBuildingBlock` with
    all intrinsics defined in `intrinsic_bodies.py` replaced with their bodies.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  context_stack = context_stack_impl.context_stack
  comp, _ = value_transformations.replace_all_intrinsics_with_bodies(
      comp, context_stack)
  return comp
