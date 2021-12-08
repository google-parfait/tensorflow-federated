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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A library of transformations that can be applied to a computation."""

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import compiled_computation_transforms


class TFParser(object):
  """Callable taking subset of TFF AST constructs to CompiledComputations.

  When this function is applied via `transformation_utils.transform_postorder`
  to a TFF AST node satisfying its assumptions, the tree under this node will
  be reduced to a single instance of `building_blocks.CompiledComputation`
  representing the same logic.

  Notice that this function is designed to be applied to what is essentially
  a subtree of a larger TFF AST; once the processing on a single device has
  been aligned at the AST level, and placement separated from the logic of
  this processing, we should be left with a function wrapped via
  `federated_map` or `federated_apply` to a federated argument. It is this
  function which we need to reduce to TensorFlow, and it is to the root
  node of this function which we are looking to apply `TFParser`. Because of
  this, we assume that there is a lambda expression at the top of the AST
  we are looking to parse, as well as the rest of the assumptions below.

  1. All called lambdas have been converted to blocks.
  2. All blocks have been inlined; that is, there are no block/LET constructs
     remaining.
  3. All compiled computations are called.
  4. No compiled computations have been partially called; we believe this
     should be handled correctly today but we haven't reasoned explicitly about
     this possibility.
  5. The only leaf nodes present under `comp` are compiled computations and
     references to the argument of the top-level lambda which we are hoping to
     replace with a compiled computation. Further, every leaf node which is a
     reference has as its parent a `building_blocks.Call`, whose
     associated function is a TF graph. This prevents us from needing to
     deal with arbitrary nesting of references and TF graphs, and significantly
     clarifies the reasoning. This can be accomplished by "decorating" the
     appropriate leaves with called identity TF graphs, the construction of
     which is provided by a utility module.
  6. There is only a single lambda binding any references present in the AST,
     and it is placed at the root of the AST to which we apply `TFParser`.
  7. There are no intrinsics present in the AST.
  """

  def __init__(self):
    """Populates the parser library with mutually exclusive options."""
    self._parse_library = [
        compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(),
        compiled_computation_transforms.LambdaWrappingGraph(),
        compiled_computation_transforms.LambdaWrappingNoArgGraph(),
        compiled_computation_transforms.StructCalledGraphs(),
        compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(),
        compiled_computation_transforms.CalledGraphOnReplicatedArg(),
    ]

  def __call__(self, comp):
    """Transforms `comp` by checking all elements of the parser library.

    This function is roughly performing intermediate-code generation, taking
    TFF and generating TF. Calling this function is essentially checking the
    stack and selecting a semantic action based on its contents, and *only one*
    of these actions should be selected for a given computation.

    Notice that since the parser library contains mutually exclusive options,
    it is safe to return early.

    Args:
      comp: The `building_blocks.ComputationBuildingBlock` to check for
        possibility of reduction according to the parsing library.

    Returns:
      A tuple whose first element is a possibly transformed version of `comp`,
      and whose second is a Boolean indicating whether or not `comp` was
      transformed. This is in conforming to the conventions of
      `transformation_utils.transform_postorder`.
    """
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    modified = False
    for option in self._parse_library:
      if option.should_transform(comp):
        comp, inner_modified = option.transform(comp)
        modified = modified or inner_modified
    return comp, modified
