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

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy

from tensorflow_federated.python.research.gans import gan_losses
from tensorflow_federated.python.research.gans import gan_training_tf_fns
from tensorflow_federated.python.research.gans import one_dim_gan
from tensorflow_federated.python.research.gans import tff_gans

# These 'before' and 'after' values bear no significance, they meerly correspond
# to the clip and std dev values before and after two rounds of training when
# all other hyperparams are set as they are. These values are asserted so as to
# provide a check for code changes that alter the behavior of the TFF GAN.
BEFORE_DP_L2_NORM_CLIP = 100.0
AFTER_2_RDS_DP_L2_NORM_CLIP = 100.4799957

BEFORE_DP_STD_DEV = 30.0000019
AFTER_2_RDS_DP_STD_DEV = 30.1439991

UPDATE_DP_L2_NORM_CLIP = 500.0


def _get_gan(with_dp=False):
  gan_loss_fns = gan_losses.get_gan_loss_fns('wasserstein')
  server_gen_optimizer = tf.keras.optimizers.Adam()
  client_disc_optimizer = tf.keras.optimizers.Adam()
  train_generator_fn = gan_training_tf_fns.create_train_generator_fn(
      gan_loss_fns, server_gen_optimizer)
  train_discriminator_fn = gan_training_tf_fns.create_train_discriminator_fn(
      gan_loss_fns, client_disc_optimizer)

  if with_dp:
    dp_average_query = tensorflow_privacy.QuantileAdaptiveClipAverageQuery(
        initial_l2_norm_clip=BEFORE_DP_L2_NORM_CLIP,
        noise_multiplier=0.3,
        target_unclipped_quantile=3,
        learning_rate=0.1,
        clipped_count_stddev=0.0,
        expected_num_records=10,
        denominator=10.0)
  else:
    dp_average_query = None

  return tff_gans.GanFnsAndTypes(
      generator_model_fn=one_dim_gan.create_generator,
      discriminator_model_fn=one_dim_gan.create_discriminator,
      dummy_gen_input=next(iter(one_dim_gan.create_generator_inputs())),
      dummy_real_data=next(iter(one_dim_gan.create_real_data())),
      train_generator_fn=train_generator_fn,
      train_discriminator_fn=train_discriminator_fn,
      server_disc_update_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=1.0),
      train_discriminator_dp_average_query=dp_average_query)


class TffGansTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('no_dp', False), ('dp', True))
  def test_build_server_initial_state_comp(self, with_dp):
    gan = _get_gan(with_dp)
    initial_state_comp = tff_gans.build_server_initial_state_comp(gan)

    # Note: Because this is a tf_computation, we preserve the Python
    # container types; this means we end up with different container
    # types than those ServerState.from_tff_result gives us, here
    # and in several places. See to-do on ServerState.from_tff_result.
    server_state = initial_state_comp()

    # Validate the initial state of the server counters.
    self.assertDictEqual(
        server_state.counters, {
            'num_rounds': 0,
            'num_generator_train_examples': 0,
            'num_discriminator_train_examples': 0
        })

    if with_dp:
      # Check DP averaging aggregation initial state is correct.
      dp_averaging_state = server_state.dp_averaging_state
      self.assertAlmostEqual(
          dp_averaging_state.numerator_state.sum_state.l2_norm_clip,
          BEFORE_DP_L2_NORM_CLIP)
      self.assertAlmostEqual(
          dp_averaging_state.numerator_state.sum_state.stddev,
          BEFORE_DP_STD_DEV)

  @parameterized.named_parameters(('no_dp', False), ('dp', True))
  def test_client_computation(self, with_dp):
    gan = _get_gan(with_dp)
    client_comp = tff_gans.build_client_computation(gan)

    generator = gan.generator_model_fn()
    discriminator = gan.discriminator_model_fn()

    from_server = gan_training_tf_fns.FromServer(
        generator_weights=generator.weights,
        discriminator_weights=discriminator.weights)
    client_output = client_comp(one_dim_gan.create_generator_inputs().take(10),
                                one_dim_gan.create_real_data().take(10),
                                from_server)
    self.assertDictEqual(
        client_output.counters,
        {'num_discriminator_train_examples': 10 * one_dim_gan.BATCH_SIZE})

  @parameterized.named_parameters(('no_dp', False), ('dp', True))
  def test_server_computation(self, with_dp):
    gan = _get_gan(with_dp)
    initial_state_comp = tff_gans.build_server_initial_state_comp(gan)

    # TODO(b/131700944): Remove this workaround, and directly instantiate a
    # ClientOutput instance (once TFF has a utility to infer TFF types of
    # objects directly).
    @tff.tf_computation
    def client_output_fn():
      discriminator = gan.discriminator_model_fn()
      return gan_training_tf_fns.ClientOutput(
          discriminator_weights_delta=[
              tf.zeros(shape=v.shape, dtype=v.dtype)
              for v in discriminator.weights
          ],
          update_weight=1.0,
          counters={'num_discriminator_train_examples': 13})

    def _update_dp_averaging_state(with_dp, dp_averaging_state):
      if not with_dp:
        return dp_averaging_state
      new_dp_averaging_state = dp_averaging_state._asdict(recursive=True)
      new_dp_averaging_state['numerator_state']['sum_state']['l2_norm_clip'] = (
          UPDATE_DP_L2_NORM_CLIP)
      return new_dp_averaging_state

    server_comp = tff_gans.build_server_computation(
        gan, initial_state_comp.type_signature.result,
        client_output_fn.type_signature.result)

    server_state = initial_state_comp()

    client_output = client_output_fn()
    new_dp_averaging_state = _update_dp_averaging_state(
        with_dp, server_state.dp_averaging_state)
    final_server_state = server_comp(
        server_state,
        one_dim_gan.create_generator_inputs().take(7), client_output,
        new_dp_averaging_state)

    # Check that server counters have incremented (compare before and after).
    self.assertDictEqual(
        server_state.counters, {
            'num_rounds': 0,
            'num_generator_train_examples': 0,
            'num_discriminator_train_examples': 0
        })
    self.assertDictEqual(
        final_server_state.counters, {
            'num_rounds': 1,
            'num_discriminator_train_examples': 13,
            'num_generator_train_examples': 7 * one_dim_gan.BATCH_SIZE
        })

    if with_dp:
      # Check that DP averaging aggregator state reflects the new state that was
      # passed as argument to server computation (compare before and after).
      initial_dp_averaging_state = server_state.dp_averaging_state
      self.assertAlmostEqual(
          initial_dp_averaging_state.numerator_state.sum_state.l2_norm_clip,
          BEFORE_DP_L2_NORM_CLIP)
      new_dp_averaging_state = final_server_state.dp_averaging_state
      self.assertAlmostEqual(
          new_dp_averaging_state.numerator_state.sum_state.l2_norm_clip,
          UPDATE_DP_L2_NORM_CLIP)

  @parameterized.named_parameters(('no_dp', False), ('dp', True))
  def test_build_gan_training_process(self, with_dp):
    gan = _get_gan(with_dp)
    process = tff_gans.build_gan_training_process(gan)
    server_state = gan_training_tf_fns.ServerState.from_tff_result(
        process.initialize())

    if with_dp:
      # Check that initial DP averaging aggregator state is correct.
      dp_averaging_state = server_state.dp_averaging_state
      self.assertAlmostEqual(
          dp_averaging_state['numerator_state']['sum_state']['l2_norm_clip'],
          BEFORE_DP_L2_NORM_CLIP)
      self.assertAlmostEqual(
          dp_averaging_state['numerator_state']['sum_state']['stddev'],
          BEFORE_DP_STD_DEV)

    client_dataset_sizes = [1, 3]
    client_gen_inputs = [
        one_dim_gan.create_generator_inputs().take(i)
        for i in client_dataset_sizes
    ]

    client_real_inputs = [
        one_dim_gan.create_real_data().take(i) for i in client_dataset_sizes
    ]

    num_rounds = 2
    for _ in range(num_rounds):
      server_state = process.next(server_state,
                                  one_dim_gan.create_generator_inputs().take(1),
                                  client_gen_inputs, client_real_inputs)

    # TODO(b/123092620): Won't need to convert from AnonymousTuple, eventually.
    server_state = gan_training_tf_fns.ServerState.from_tff_result(server_state)

    # Check that server counters have incremented.
    counters = server_state.counters
    self.assertDictEqual(
        counters, {
            'num_rounds':
                num_rounds,
            'num_generator_train_examples':
                one_dim_gan.BATCH_SIZE * num_rounds,
            'num_discriminator_train_examples':
                num_rounds * one_dim_gan.BATCH_SIZE * sum(client_dataset_sizes),
        })

    if with_dp:
      # Check that DP averaging aggregator state has updated properly over the
      # above rounds.
      dp_averaging_state = server_state.dp_averaging_state
      self.assertAlmostEqual(
          dp_averaging_state['numerator_state']['sum_state']['l2_norm_clip'],
          AFTER_2_RDS_DP_L2_NORM_CLIP)
      self.assertAlmostEqual(
          dp_averaging_state['numerator_state']['sum_state']['stddev'],
          AFTER_2_RDS_DP_STD_DEV)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
