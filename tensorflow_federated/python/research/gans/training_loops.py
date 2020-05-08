# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""TF and TFF training loops."""

import os.path
import time

from absl import logging
import attr
import tensorflow as tf

from tensorflow_federated.python.research.gans import gan_training_tf_fns
from tensorflow_federated.python.research.gans import tff_gans
from tensorflow_federated.python.research.utils import checkpoint_utils

CHECKPOINT_PREFIX = 'ckpt_'


def simple_training_loop(generator_model_fn,
                         discriminator_model_fn,
                         real_data_fn,
                         gen_inputs_fn,
                         train_generator_fn,
                         train_discriminator_fn,
                         total_rounds=30,
                         client_disc_train_steps=16,
                         server_gen_train_steps=8,
                         rounds_per_eval=10,
                         eval_hook=lambda *args: None):
  """Trains in TF using client_computation and server_computation.

  This is not intended to be a general-purpose training loop (e.g., the
  optimizers are hard-coded), it is primarily intended for testing.

  Args:
    generator_model_fn: A no-arg function return the generator model.
    discriminator_model_fn: A no-arg function return the discriminator model.
    real_data_fn: A no-arg function returning a dataset of real data batches.
    gen_inputs_fn: A no-arg function returning a dataset of generator input
      batches.
    train_generator_fn: A function which takes the two networks and generator
      input and trains the generator.
    train_discriminator_fn: A function which takes the two networks, generator
      input, and real data and trains the discriminator.
    total_rounds: Number of rounds to train.
    client_disc_train_steps: Number of discriminator training batches per round.
    server_gen_train_steps: Number of generator training batches per round.
    rounds_per_eval: How often to call the  `eval_hook` function.
    eval_hook: A function taking arguments (generator, discriminator,
      server_state, round_num) and performs evaluation. Optional.

  Returns:
    A tuple (final `ServerState`, train_time_in_seconds).
  """
  logging.info('Starting simple_training_loop')
  # N.B. We can't use real_data.take(...) in the loops below,
  # or we would get the same examples on every round. Using window
  # essentially breaks one Dataset into a sequence of Datasets,
  # which is exactly what we need here.
  client_gen_inputs = iter(gen_inputs_fn().window(client_disc_train_steps))
  client_real_data = iter(real_data_fn().window(client_disc_train_steps))

  server_gen_inputs = iter(gen_inputs_fn().window(server_gen_train_steps))

  server_generator = generator_model_fn()
  server_discriminator = discriminator_model_fn()
  # We could probably use a single copy of the generator and discriminator, but
  # using separate copies is more faithful to how this code will be used in TFF.
  client_generator = generator_model_fn()
  client_discriminator = discriminator_model_fn()

  server_disc_update_optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

  server_state = gan_training_tf_fns.server_initial_state(
      server_generator, server_discriminator)

  start_time = time.time()

  def do_eval(round_num):
    eval_hook(server_generator, server_discriminator, server_state, round_num)
    elapsed_minutes = (time.time() - start_time) / 60
    print(
        'Total training time {:.2f} minutes for {} rounds '
        '({:.2f} rounds per minute)'.format(elapsed_minutes, round_num,
                                            round_num / elapsed_minutes),
        flush=True)

  logging.info('Starting training')
  for round_num in range(total_rounds):
    if round_num % rounds_per_eval == 0:
      do_eval(round_num)

    client_output = gan_training_tf_fns.client_computation(
        gen_inputs_ds=next(client_gen_inputs),
        real_data_ds=next(client_real_data),
        from_server=gan_training_tf_fns.FromServer(
            generator_weights=server_state.generator_weights,
            discriminator_weights=server_state.discriminator_weights),
        generator=client_generator,
        discriminator=client_discriminator,
        train_discriminator_fn=train_discriminator_fn)

    server_state = gan_training_tf_fns.server_computation(
        server_state=server_state,
        gen_inputs_ds=next(server_gen_inputs),
        client_output=client_output,
        generator=server_generator,
        discriminator=server_discriminator,
        server_disc_update_optimizer=server_disc_update_optimizer,
        train_generator_fn=train_generator_fn)

  train_time = time.time() - start_time
  do_eval(total_rounds)
  return server_state, train_time


@attr.s()
class ExperimentState(object):
  """Container from the state of a federated_training_loop."""
  # Note: This is used only inside read_checkpoint and write_checkpoint, where
  # any marshalling/unmarshalling logic currently lives. An alternative design
  # makes these methods essentially generic (likely moving to checkpoint_utils),
  # with an object of this type (where all fields are already marshalled) passed
  # in.

  round_num = attr.ib()
  # Server state contains the generator and discriminator weights, the counters,
  # and the state of the DP averaging aggregation, so all data needed for
  # restoring and continuing.
  server_state = attr.ib()

  # Note: Could add EvalHook state here if needed, though would need
  # to say have EvalHook be an object with a __call__ method, but also say
  # get_state and from_state methods.


def write_checkpoint(root_checkpoint_dir, server_state, round_num):
  """Write the current experiment state to disk."""
  if root_checkpoint_dir is None:
    return

  if tf.io.gfile.exists(root_checkpoint_dir):
    # Clean-up old checkpoints if more than 5 exist, not including the
    # original (which captures random model initialization).
    checkpoints = sorted(
        tf.io.gfile.glob(
            os.path.join(root_checkpoint_dir, CHECKPOINT_PREFIX + '*')))
    to_remove = checkpoints[1:-1]
    logging.info('Cleaning up %s', to_remove)
    for checkpoint in to_remove:
      tf.io.gfile.rmtree(checkpoint)

  state = ExperimentState(round_num, server_state)
  checkpoint_dir = os.path.join(root_checkpoint_dir,
                                '{}{:04d}'.format(CHECKPOINT_PREFIX, round_num))
  checkpoint_utils.save(state, checkpoint_dir)


def read_checkpoint(checkpoint_dir, server_state):
  """Read a previously saved experiment state to memory."""
  obj_template = ExperimentState(round_num=0, server_state=server_state)
  state = checkpoint_utils.load(checkpoint_dir, obj_template)

  return state.server_state, state.round_num


def maybe_read_latest_checkpoint(root_checkpoint_dir, server_state):
  """Returns server_state, round_num, possibly from a recent checkpoint."""
  if root_checkpoint_dir is None:
    latest_checkpoint_dir = None
  else:
    latest_checkpoint_dir = checkpoint_utils.latest_checkpoint(
        root_checkpoint_dir, CHECKPOINT_PREFIX)
    logging.info('Looking for checkpoints in [%s/%s].', root_checkpoint_dir,
                 CHECKPOINT_PREFIX)
  if latest_checkpoint_dir is None:
    write_checkpoint(root_checkpoint_dir, server_state, 0)
    logging.info('No previous checkpoints, initializing experiment.')
    return server_state, 0
  else:
    server_state, round_num = read_checkpoint(latest_checkpoint_dir,
                                              server_state)
    round_num = int(round_num.numpy())
    logging.info('Restarting from checkpoint round %d.', round_num)
    return server_state, round_num


def federated_training_loop(gan: tff_gans.GanFnsAndTypes,
                            server_gen_inputs_fn,
                            client_datasets_fn,
                            total_rounds=1,
                            rounds_per_eval=1,
                            eval_hook=lambda *args: None,
                            rounds_per_checkpoint=5,
                            root_checkpoint_dir=None):
  """A simple federated training loop.

  Args:
    gan: A `GanFnsAndTypes` object.
    server_gen_inputs_fn: A function that takes the round number, and returns a
      dataset of generator inputs to use to train the generator for this round.
    client_datasets_fn: A function that takes the round number, and returns a
      list of tuples of (gen_inputs, real_data) Datasets, one per client.
    total_rounds: Number of rounds to train.
    rounds_per_eval: How often to call the  `eval_hook` function.
    eval_hook: A function taking arguments (generator, discriminator,
      server_state, round_num) and performs evaluation. Optional.
    rounds_per_checkpoint: If root_checkpoint_dir is given, how often to
      checkpoint.
    root_checkpoint_dir: Optional directory from which read and write
      checkpoints.  If execution is interupted and restarted, these checkpoints
      can be used for restart where things left off.

  Returns:
    A tuple (final `ServerState`, train_time_in_seconds).
  """
  logging.info('Starting federated_training_loop.')
  start_time = time.time()
  process = tff_gans.build_gan_training_process(gan)
  # TODO(b/123092620): The following conversion (from anon tuple to ServerState)
  # should not be needed.
  server_state = gan_training_tf_fns.ServerState.from_tff_result(
      process.initialize())
  logging.info(
      'Built processes and computed initial state in {:.2f} seconds'.format(
          time.time() - start_time))

  server_state, round_num = maybe_read_latest_checkpoint(
      root_checkpoint_dir, server_state)

  start_time = time.time()
  start_round_num = round_num

  eval_generator = gan.generator_model_fn()
  eval_discriminator = gan.discriminator_model_fn()

  def update_eval_models(server_state):
    eval_generator.set_weights(server_state.generator_weights)
    eval_discriminator.set_weights(server_state.discriminator_weights)

  def do_eval(round_num, server_state):
    update_eval_models(server_state)
    eval_hook(eval_generator, eval_discriminator, server_state, round_num)
    elapsed_minutes = (time.time() - start_time) / 60
    elapsed_rounds = round_num - start_round_num
    print(
        'Round #{}. Total training time {:.2f} minutes for {} rounds '
        '({:.2f} rounds per minute)'.format(round_num, elapsed_minutes,
                                            elapsed_rounds,
                                            elapsed_rounds / elapsed_minutes),
        flush=True)

  logging.info('Starting training.')
  while round_num < total_rounds:
    if round_num % rounds_per_eval == 0:
      do_eval(round_num, server_state)

    client_gen_inputs, client_real_inputs = zip(*client_datasets_fn(round_num))
    # TODO(b/123092620): The following conversion (from anon tuple to
    # ServerState) should not be needed.
    server_state = gan_training_tf_fns.ServerState.from_tff_result(
        process.next(server_state, server_gen_inputs_fn(round_num),
                     client_gen_inputs, client_real_inputs))

    round_num += 1
    if round_num % rounds_per_checkpoint == 0:
      write_checkpoint(root_checkpoint_dir, server_state, round_num)

  train_time = time.time() - start_time
  do_eval(total_rounds, server_state)
  return server_state, train_time
