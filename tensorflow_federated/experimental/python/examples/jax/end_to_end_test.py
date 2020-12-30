# Copyright 2020, The TensorFlow Federated Authors.
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
import jax
import numpy as np
import tensorflow_federated as tff


class EndToEndTest(absltest.TestCase):

  # TODO(b/175888145): Evolve this into a complete federated training example.

  def test_add_numbers(self):

    @tff.experimental.jax_computation(np.int32, np.int32)
    def foo(x, y):
      return jax.numpy.add(x, y)

    result = foo(np.int32(20), np.int32(30))
    self.assertEqual(result, 50)


if __name__ == '__main__':
  tff.experimental.backends.xla.set_local_execution_context()
  absltest.main()
