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
from absl.testing import absltest
import jax
import numpy as np
import tensorflow_federated as tff

# TODO(b/175888145): Represent the entirety of local eval as a computation.

# TODO(b/175888145): Evolve this into a complete federated training example.

# TODO(b/175888145): Allow step size to be dynamic.

Trainer = collections.namedtuple('Trainer', [
    'create_initial_model',
    'generate_random_batches',
    'train_on_one_batch',
    'train_on_one_client',
    'local_training_process',
    'compute_loss_on_one_batch',
])


def create_trainer(batch_size, step_size):
  """Constructs a trainer for the given batch size.

  Args:
    batch_size: The size of a single data batch.
    step_size: The step size to use during training.

  Returns:
    An instance of `Trainer`.
  """
  batch_type = tff.to_type(
      collections.OrderedDict([
          ('pixels', tff.TensorType(np.float32, (batch_size, 784))),
          ('labels', tff.TensorType(np.int32, (batch_size,)))
      ]))

  model_type = tff.to_type(
      collections.OrderedDict([('weights',
                                tff.TensorType(np.float32, (784, 10))),
                               ('bias', tff.TensorType(np.float32, (10,)))]))

  @tff.experimental.jax_computation
  def create_initial_model():
    weights = jax.numpy.zeros((784, 10), dtype=np.float32)
    bias = jax.numpy.zeros((10,), dtype=np.float32)
    return collections.OrderedDict([('weights', weights), ('bias', bias)])

  def generate_random_batches(num_batches):
    for _ in range(num_batches):
      pixels = np.random.uniform(
          low=0.0, high=1.0, size=(batch_size, 784)).astype(np.float32)
      labels = np.random.randint(
          low=0, high=9, size=(batch_size,), dtype=np.int32)
      yield collections.OrderedDict([('pixels', pixels), ('labels', labels)])

  def _loss_fn(model, batch):
    y = jax.nn.softmax(
        jax.numpy.add(
            jax.numpy.matmul(batch['pixels'], model['weights']), model['bias']))
    targets = jax.nn.one_hot(jax.numpy.reshape(batch['labels'], -1), 10)
    return -jax.numpy.mean(jax.numpy.sum(targets * jax.numpy.log(y), axis=1))

  @tff.experimental.jax_computation(model_type, batch_type)
  def train_on_one_batch(model, batch):
    grads = jax.api.grad(_loss_fn)(model, batch)
    return collections.OrderedDict([
        (k, model[k] - step_size * grads[k]) for k in ['weights', 'bias']
    ])

  @tff.federated_computation(model_type, tff.SequenceType(batch_type))
  def train_on_one_client(model, batches):
    return tff.sequence_reduce(batches, model, train_on_one_batch)

  local_training_process = tff.templates.IterativeProcess(
      initialize_fn=create_initial_model, next_fn=train_on_one_client)

  compute_loss_on_one_batch = tff.experimental.jax_computation(
      _loss_fn, model_type, batch_type)

  return Trainer(
      create_initial_model=create_initial_model,
      generate_random_batches=generate_random_batches,
      train_on_one_batch=train_on_one_batch,
      train_on_one_client=train_on_one_client,
      local_training_process=local_training_process,
      compute_loss_on_one_batch=compute_loss_on_one_batch)


class JaxTrainingTest(absltest.TestCase):

  def test_types(self):
    trainer = create_trainer(100, 0.01)
    model_type = trainer.create_initial_model.type_signature.result
    example_batch = next(trainer.generate_random_batches(1))
    make_example_batch = tff.experimental.jax_computation(lambda: example_batch)
    batch_type = make_example_batch.type_signature.result
    self.assertEqual(
        str(trainer.train_on_one_batch.type_signature),
        str(
            tff.FunctionType(
                collections.OrderedDict([('model', model_type),
                                         ('batch', batch_type)]), model_type)))
    self.assertEqual(
        str(trainer.compute_loss_on_one_batch.type_signature),
        str(
            tff.FunctionType(
                collections.OrderedDict([('model', model_type),
                                         ('batch', batch_type)]), np.float32)))

  def test_training(self):
    batch_size = 50
    num_batches = 20
    num_rounds = 5
    step_size = 0.001
    trainer = create_trainer(batch_size, step_size)
    training_batches = list(trainer.generate_random_batches(num_batches))
    eval_batches = list(trainer.generate_random_batches(num_batches))

    model = trainer.local_training_process.initialize()
    losses = []
    for round_number in range(num_rounds + 1):
      if round_number > 0:
        model = trainer.local_training_process.next(model, training_batches)
      average_loss = np.mean([
          trainer.compute_loss_on_one_batch(model, batch)
          for batch in eval_batches
      ])
      losses.append(average_loss)
    self.assertLess(losses[-1], losses[0])


if __name__ == '__main__':
  tff.experimental.backends.xla.set_local_execution_context()
  absltest.main()
