# Copyright 2021, The TensorFlow Federated Authors.
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

from absl.testing import absltest
import numpy as np

from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class DesugarAndTransformTest(absltest.TestCase):

  def test_desugaring_sum_insert_id_for_tf_computations(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def fed_sum(x):
      return intrinsics.federated_sum(x)

    reduced_comp = compiler.desugar_and_transform_to_native(fed_sum)
    reduced_bb = reduced_comp.to_building_block()

    def _check_tf_computations_have_ids(comp):
      if (
          isinstance(comp, building_blocks.CompiledComputation)
          and comp.proto.WhichOneof('computation') == 'tensorflow'
          and not comp.proto.tensorflow.cache_key.id
      ):
        raise ValueError(
            f'Building block {comp.formatted_representation()} is a compiled '
            'computation of TensorFlow type but without a cache key ID.'
        )
      return comp, False

    # Doesn't raise.
    transformation_utils.transform_postorder(
        reduced_bb, _check_tf_computations_have_ids
    )


if __name__ == '__main__':
  absltest.main()
