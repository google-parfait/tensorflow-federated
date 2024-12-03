## Copyright 2022, The TensorFlow Federated Authors.
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
"""Utilities for testing compiler."""

import collections

import federated_language

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.common_libs import py_typecheck


# Name the compiled computations to avoid the issue that the TF graphs being
# generated are different at HEAD vs in OSS, resulting in different hash values
# for the computation name which fail to compare.
def _name_compiled_computations(
    tree: federated_language.framework.ComputationBuildingBlock,
) -> federated_language.framework.ComputationBuildingBlock:
  """Name the compiled computations."""
  counter = 1

  def _transform(building_block):
    nonlocal counter
    if isinstance(
        building_block, federated_language.framework.CompiledComputation
    ):
      new_name = str(counter)
      counter += 1
      return (
          federated_language.framework.CompiledComputation(
              proto=building_block.proto, name=new_name
          ),
          True,
      )
    return building_block, False

  return federated_language.framework.transform_postorder(tree, _transform)[0]


def check_computations(
    filename: str,
    computations: collections.OrderedDict[
        str, federated_language.framework.ComputationBuildingBlock
    ],
) -> None:
  """Check the AST of computations matches the contents of the golden file.

  Args:
    filename: String filename of the golden file.
    computations: An `collections.OrderedDict` of computation names to
      `federated_language.framework.ComputationBuildingBlock`.

  Raises:
    TypeError: If any argument type mismatches.
  """
  py_typecheck.check_type(filename, str)
  py_typecheck.check_type(computations, collections.OrderedDict, 'computations')
  values = []
  for name, computation in computations.items():
    py_typecheck.check_type(
        computation, federated_language.framework.ComputationBuildingBlock, name
    )
    computation_ast = _name_compiled_computations(computation)
    values.append(
        f'{name}:\n\n{computation_ast.formatted_representation()}\n\n'
    )
  golden.check_string(filename, ''.join(values))
