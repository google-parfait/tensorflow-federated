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
"""End-to-end test of training in JAX.

This test is still evolving. Eventually, it should do federated training, run
with a model of sufficient complexity, etc., and morph into a form that we could
consider moving out of experimental and capturing in a tutorial. For the time
beingm, this test is here to ensure that we don't break JAX support as we evolve
it to be more complete and feature-full.
"""

import collections
import itertools
from absl.testing import absltest
import jax
import numpy as np
import tensorflow_federated as tff


BATCH_TYPE = collections.OrderedDict([
    ('pixels', tff.TensorType(np.float32, (50, 784))),
    ('labels', tff.TensorType(np.int32, (50,)))
])

MODEL_TYPE = collections.OrderedDict([
    ('weights', tff.TensorType(np.float32, (784, 10))),
    ('bias', tff.TensorType(np.float32, (10,)))
])


def loss(model, batch):
  y = jax.nn.softmax(
      jax.numpy.add(
          jax.numpy.matmul(batch['pixels'], model['weights']), model['bias']))
  targets = jax.nn.one_hot(jax.numpy.reshape(batch['labels'], -1), 10)
  return -jax.numpy.mean(jax.numpy.sum(targets * jax.numpy.log(y), axis=1))


def prepare_data(num_clients, num_batches):
  federated_training_data = []
  for _ in range(num_clients):
    batches = []
    for _ in range(num_batches):
      pixels = np.random.uniform(
          low=0.0, high=1.0, size=(50, 784)).astype(np.float32)
      labels = np.random.randint(low=0, high=9, size=(50,), dtype=np.int32)
      batch = collections.OrderedDict([('pixels', pixels), ('labels', labels)])
      batches.append(batch)
    federated_training_data.append(batches)
  centralized_eval_data = list(
      itertools.chain.from_iterable(federated_training_data))
  return federated_training_data, centralized_eval_data


class JaxTrainingTest(absltest.TestCase):

  def test_federated_training(self):
    training_data, eval_data = prepare_data(num_clients=2, num_batches=10)
    trainer = tff.experimental.learning.build_jax_federated_averaging_process(
        BATCH_TYPE, MODEL_TYPE, loss, step_size=0.001)
    model = trainer.initialize()
    losses = []
    num_rounds = 5
    for round_number in range(num_rounds + 1):
      if round_number > 0:
        model = trainer.next(model, training_data)
      average_loss = np.mean([loss(model, batch) for batch in eval_data])
      losses.append(average_loss)
    self.assertLess(losses[-1], losses[0])


if __name__ == '__main__':
  tff.backends.xla.set_local_execution_context()
  absltest.main()
