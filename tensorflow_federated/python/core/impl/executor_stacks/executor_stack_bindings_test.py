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
from absl.testing import parameterized
import numpy as np

from pybind11_abseil import status as absl_status
from tensorflow_federated.python.core.impl.executor_stacks import executor_stack_bindings
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.types import placements

_TARGET_LIST = ['localhost:8000', 'localhost:8001']
_CARDINALITIES = {placements.CLIENTS: 5}


class ExecutorStackBindingsTest(parameterized.TestCase):

  @parameterized.named_parameters(('from_target_list', list),
                                  ('from_target_tuple', tuple),
                                  ('from_target_ndarray', np.array))
  def test_executor_construction_raises_no_channels_available(
      self, container_constructor):
    with self.assertRaisesRegex(absl_status.StatusNotOk, 'UNAVAILABLE'):
      executor_stack_bindings.create_remote_executor_stack(
          channels=container_constructor([
              executor_bindings.create_insecure_grpc_channel(t)
              for t in _TARGET_LIST
          ]),
          cardinalities=_CARDINALITIES)


if __name__ == '__main__':
  absltest.main()
