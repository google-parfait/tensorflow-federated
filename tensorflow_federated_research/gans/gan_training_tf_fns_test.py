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

import tensorflow as tf

from tensorflow_federated_research.gans import gan_losses
from tensorflow_federated_research.gans import gan_training_tf_fns
from tensorflow_federated_research.gans import one_dim_gan

GAN_LOSS_FNS = gan_losses.get_gan_loss_fns('wasserstein')
INIT_DP_AVERAGING_STATE = ['foo']
NEW_DP_AVERAGING_STATE = ['bar']


def _get_train_generator_and_discriminator_fns():
  train_generator_fn = gan_training_tf_fns.create_train_generator_fn(
      GAN_LOSS_FNS, tf.keras.optimizers.Adam())
  train_discriminator_fn = gan_training_tf_fns.create_train_discriminator_fn(
      GAN_LOSS_FNS, tf.keras.optimizers.Adam())
  return train_generator_fn, train_discriminator_fn


class GanTrainingTfFnsTest(tf.test.TestCase):

  def test_create_train_generator_fn(self):
    train_generator_fn = gan_training_tf_fns.create_train_generator_fn(
        GAN_LOSS_FNS, tf.keras.optimizers.Adam())
    self.assertListEqual(['generator', 'discriminator', 'generator_inputs'],
                         train_generator_fn.function_spec.fullargspec.args)

  def test_create_train_discriminator_fn(self):
    train_discriminator_fn = gan_training_tf_fns.create_train_discriminator_fn(
        GAN_LOSS_FNS, tf.keras.optimizers.Adam())
    self.assertListEqual(
        ['generator', 'discriminator', 'generator_inputs', 'real_data'],
        train_discriminator_fn.function_spec.fullargspec.args)

  def test_client_and_server_computations(self):
    train_generator_fn, train_discriminator_fn = (
        _get_train_generator_and_discriminator_fns())

    # N.B. The way we are using datasets and re-using the same
    # generator and discriminator doesn't really "make sense" from an ML
    # perspective, but it's sufficient for testing. For more proper usage of
    # these functions, see training_loops.py.
    generator = one_dim_gan.create_generator()
    discriminator = one_dim_gan.create_discriminator()
    gen_inputs = one_dim_gan.create_generator_inputs()
    real_data = one_dim_gan.create_real_data()

    server_state = gan_training_tf_fns.server_initial_state(
        generator, discriminator, INIT_DP_AVERAGING_STATE)

    # DP averaging aggregation state is initialized properly in
    # server_initial_state().
    self.assertEqual(server_state.dp_averaging_state, INIT_DP_AVERAGING_STATE)

    client_output = gan_training_tf_fns.client_computation(
        gen_inputs.take(3), real_data.take(3),
        gan_training_tf_fns.FromServer(
            generator_weights=server_state.generator_weights,
            discriminator_weights=server_state.discriminator_weights),
        generator, discriminator, train_discriminator_fn)

    server_disc_update_optimizer = tf.keras.optimizers.Adam()
    for _ in range(2):  # Train for 2 rounds
      server_state = gan_training_tf_fns.server_computation(
          server_state, gen_inputs.take(3), client_output, generator,
          discriminator, server_disc_update_optimizer, train_generator_fn,
          NEW_DP_AVERAGING_STATE)

    counters = self.evaluate(server_state.counters)
    self.assertDictEqual(
        counters, {
            'num_rounds': 2,
            'num_discriminator_train_examples': 2 * 3 * one_dim_gan.BATCH_SIZE,
            'num_generator_train_examples': 2 * 3 * one_dim_gan.BATCH_SIZE
        })

    # DP averaging aggregation state updates properly in server_computation().
    self.assertEqual(server_state.dp_averaging_state, NEW_DP_AVERAGING_STATE)


if __name__ == '__main__':
  tf.test.main()
