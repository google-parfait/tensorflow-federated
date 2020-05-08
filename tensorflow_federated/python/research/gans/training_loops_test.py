# Lint as: python3
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

import os

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
import tensorflow_privacy

from tensorflow_federated.python.research.gans import gan_losses
from tensorflow_federated.python.research.gans import gan_training_tf_fns
from tensorflow_federated.python.research.gans import one_dim_gan
from tensorflow_federated.python.research.gans import tff_gans
from tensorflow_federated.python.research.gans import training_loops


def _get_train_generator_and_discriminator_fns():
  gan_loss_fns = gan_losses.get_gan_loss_fns('wasserstein')
  train_generator_fn = gan_training_tf_fns.create_train_generator_fn(
      gan_loss_fns, tf.keras.optimizers.Adam())
  train_discriminator_fn = gan_training_tf_fns.create_train_discriminator_fn(
      gan_loss_fns, tf.keras.optimizers.Adam())
  return train_generator_fn, train_discriminator_fn


def _get_dp_average_query():
  return tensorflow_privacy.QuantileAdaptiveClipAverageQuery(
      initial_l2_norm_clip=100.0,
      noise_multiplier=0.3,
      target_unclipped_quantile=3,
      learning_rate=0.1,
      clipped_count_stddev=0.0,
      expected_num_records=10,
      denominator=10.0)


class TrainingLoopsTest(tf.test.TestCase, parameterized.TestCase):

  def test_simple_training(self):
    train_generator_fn, train_discriminator_fn = (
        _get_train_generator_and_discriminator_fns())

    server_state, _ = training_loops.simple_training_loop(
        generator_model_fn=one_dim_gan.create_generator,
        discriminator_model_fn=one_dim_gan.create_discriminator,
        real_data_fn=one_dim_gan.create_real_data,
        gen_inputs_fn=one_dim_gan.create_generator_inputs,
        train_generator_fn=train_generator_fn,
        train_discriminator_fn=train_discriminator_fn,
        total_rounds=2,
        client_disc_train_steps=2,
        server_gen_train_steps=3,
        rounds_per_eval=10)

    self.assertDictEqual(
        self.evaluate(server_state.counters), {
            'num_rounds': 2,
            'num_generator_train_examples': 2 * 3 * one_dim_gan.BATCH_SIZE,
            'num_discriminator_train_examples': 2 * 2 * one_dim_gan.BATCH_SIZE
        })

  @parameterized.named_parameters(('no_dp_and_checkpoint', None, True),
                                  ('dp', _get_dp_average_query(), False))
  def test_tff_training_loop(self, dp_average_query, checkpoint):
    if checkpoint:
      root_checkpoint_dir = os.path.join(self.get_temp_dir(), 'checkpoints')
    else:
      root_checkpoint_dir = None

    train_generator_fn, train_discriminator_fn = (
        _get_train_generator_and_discriminator_fns())

    gan = tff_gans.GanFnsAndTypes(
        generator_model_fn=one_dim_gan.create_generator,
        discriminator_model_fn=one_dim_gan.create_discriminator,
        dummy_gen_input=next(iter(one_dim_gan.create_generator_inputs())),
        dummy_real_data=next(iter(one_dim_gan.create_real_data())),
        train_generator_fn=train_generator_fn,
        train_discriminator_fn=train_discriminator_fn,
        server_disc_update_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=1.0),
        train_discriminator_dp_average_query=dp_average_query)

    gen_inputs = one_dim_gan.create_generator_inputs()
    real_data = one_dim_gan.create_real_data()

    client_disc_train_steps = 2
    server_gen_train_steps = 3

    server_gen_inputs = iter(gen_inputs.window(server_gen_train_steps))
    client_gen_inputs = iter(gen_inputs.window(client_disc_train_steps))
    client_real_data = iter(real_data.window(client_disc_train_steps))

    def server_gen_inputs_fn(_):
      return next(server_gen_inputs)

    num_clients = 2

    def client_datasets_fn(_):
      return [(next(client_gen_inputs), next(client_real_data))
              for _ in range(num_clients)]

    server_state, _ = training_loops.federated_training_loop(
        gan,
        server_gen_inputs_fn=server_gen_inputs_fn,
        client_datasets_fn=client_datasets_fn,
        total_rounds=2,
        rounds_per_checkpoint=1,
        root_checkpoint_dir=root_checkpoint_dir)

    self.assertDictEqual(
        server_state.counters, {
            'num_rounds':
                2,
            'num_generator_train_examples':
                2 * 3 * one_dim_gan.BATCH_SIZE,
            'num_discriminator_train_examples':
                (2 * 2 * one_dim_gan.BATCH_SIZE * num_clients)
        })
    if checkpoint:
      # TODO(b/141112101): We shouldn't need to re-create the gan, should be
      # able to reuse the instance from above. See comment inside tff_gans.py.
      train_generator_fn, train_discriminator_fn = (
          _get_train_generator_and_discriminator_fns())
      gan = tff_gans.GanFnsAndTypes(
          generator_model_fn=one_dim_gan.create_generator,
          discriminator_model_fn=one_dim_gan.create_discriminator,
          dummy_gen_input=next(iter(one_dim_gan.create_generator_inputs())),
          dummy_real_data=next(iter(one_dim_gan.create_real_data())),
          train_generator_fn=train_generator_fn,
          train_discriminator_fn=train_discriminator_fn,
          server_disc_update_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=1.0
                                                                         ),
          train_discriminator_dp_average_query=dp_average_query)
      # Train one more round, which should resume from the checkpoint.
      server_state, _ = training_loops.federated_training_loop(
          gan,
          server_gen_inputs_fn=server_gen_inputs_fn,
          client_datasets_fn=client_datasets_fn,
          total_rounds=3,
          rounds_per_checkpoint=1,
          root_checkpoint_dir=root_checkpoint_dir)
      # Note: It would be better to return something from
      # federated_training_loop indicating the number of rounds trained in this
      # invocation, so we could verify the checkpoint was read.
      self.assertDictEqual(
          server_state.counters, {
              'num_rounds':
                  3,
              'num_generator_train_examples':
                  3 * 3 * one_dim_gan.BATCH_SIZE,
              'num_discriminator_train_examples':
                  (3 * 2 * one_dim_gan.BATCH_SIZE * num_clients)
          })


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
