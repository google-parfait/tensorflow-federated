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

from tensorflow_federated.python.core.impl.executors import data_descriptor
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class DataDescriptorTest(absltest.TestCase):

  def test_raises_with_server_cardinality_specified(self):
    with self.assertRaises(ValueError):
      data_descriptor.DataDescriptor(
          federated_computation.federated_computation(
              lambda x: intrinsics.federated_value(x, placements.SERVER),
              np.int32,
          ),
          1000,
          computation_types.TensorType(np.int32),
          3,
      )


if __name__ == '__main__':
  # TFF-CPP does not yet speak `Ingestable`; b/202336418
  absltest.main()
