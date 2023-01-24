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
"""A compiler for the test backend."""

from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.computation import computation_impl


def replace_secure_intrinsics_with_bodies(comp):
  """Replace `secure_...` intrinsics with insecure TensorFlow equivalents.

  Designed for use in tests, this function replaces
  `tff.federated_secure_{sum, sum_bitwidth, modular_sum}` usages with equivalent
  TensorFlow computations. The resulting computation can then be run on TFF
  runtimes which do not implement secure computation.

  Args:
    comp: The computation to transform.

  Returns:
    `comp` with secure intrinsics replaced with insecure TensorFlow equivalents.
  """
  # Compile secure_sum and secure_sum_bitwidth intrinsics to insecure
  # TensorFlow computations for testing purposes.
  replaced_intrinsic_bodies, _ = (
      tree_transformations.replace_secure_intrinsics_with_insecure_bodies(
          comp.to_building_block()
      )
  )
  return computation_impl.ConcreteComputation.from_building_block(
      replaced_intrinsic_bodies
  )
