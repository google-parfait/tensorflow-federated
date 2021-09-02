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

import collections

from absl.testing import absltest
import jax
import numpy as np

from tensorflow_federated.experimental.python.learning import jax_components
from tensorflow_federated.python.core.backends.xla import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types


class JaxComponentsTest(absltest.TestCase):

  def test_build_jax_federated_averaging_process(self):
    batch_type = collections.OrderedDict([
        ('pixels', computation_types.TensorType(np.float32, (50, 784))),
        ('labels', computation_types.TensorType(np.int32, (50,)))
    ])

    def random_batch():
      pixels = np.random.uniform(
          low=0.0, high=1.0, size=(50, 784)).astype(np.float32)
      labels = np.random.randint(low=0, high=9, size=(50,), dtype=np.int32)
      return collections.OrderedDict([('pixels', pixels), ('labels', labels)])

    model_type = collections.OrderedDict([
        ('weights', computation_types.TensorType(np.float32, (784, 10))),
        ('bias', computation_types.TensorType(np.float32, (10,)))
    ])

    def loss(model, batch):
      y = jax.nn.softmax(
          jax.numpy.add(
              jax.numpy.matmul(batch['pixels'], model['weights']),
              model['bias']))
      targets = jax.nn.one_hot(jax.numpy.reshape(batch['labels'], -1), 10)
      return -jax.numpy.mean(jax.numpy.sum(targets * jax.numpy.log(y), axis=1))

    trainer = jax_components.build_jax_federated_averaging_process(
        batch_type, model_type, loss, step_size=0.001)

    trainer.next(trainer.initialize(), [[random_batch()]])


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  absltest.main()
