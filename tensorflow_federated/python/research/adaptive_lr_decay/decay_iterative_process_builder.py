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
"""Library for building federated averaging processes with LR decay from flags."""

from absl import flags
import tensorflow_federated as tff

from tensorflow_federated.python.research.adaptive_lr_decay import adaptive_fed_avg
from tensorflow_federated.python.research.adaptive_lr_decay import callbacks
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.utils import utils_impl

# Defining optimizer flags
with utils_impl.record_hparam_flags():
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

  flags.DEFINE_float(
      'client_decay_factor', 0.1, 'Amount to decay the client learning rate '
      'upon reaching a plateau.')
  flags.DEFINE_float(
      'server_decay_factor', 0.9, 'Amount to decay the server learning rate '
      'upon reaching a plateau.')
  flags.DEFINE_float(
      'min_delta', 1e-4, 'Minimum delta for improvement in the learning rate '
      'callbacks.')
  flags.DEFINE_integer(
      'window_size', 100, 'Number of rounds to take a moving average over when '
      'estimating the training loss in learning rate callbacks.')
  flags.DEFINE_integer(
      'patience', 100, 'Number of rounds of non-improvement before decaying the'
      'learning rate.')
  flags.DEFINE_float('min_lr', 0.0, 'The minimum learning rate.')

FLAGS = flags.FLAGS


def from_flags(input_spec,
               model_builder,
               loss_builder,
               metrics_builder,
               client_weight_fn=None):
  """Builds a `tff.templates.IterativeProcess` instance from flags.

  The iterative process is designed to incorporate learning rate schedules,
  which are configured via flags.

  Args:
    input_spec: A value convertible to a `tff.Type`, representing the data which
      will be fed into the `tff.templates.IterativeProcess.next` function over
      the course of training. Generally, this can be found by accessing the
      `element_spec` attribute of a client `tf.data.Dataset`.
    model_builder: A no-arg function that returns an uncompiled `tf.keras.Model`
      object.
    loss_builder: A no-arg function returning a `tf.keras.losses.Loss` object.
    metrics_builder: A no-arg function that returns a list of
      `tf.keras.metrics.Metric` objects.
    client_weight_fn: An optional callable that takes the result of
      `tff.learning.Model.report_local_outputs` from the model returned by
      `model_builder`, and returns a scalar client weight. If `None`, defaults
      to the number of examples processed over all batches.

  Returns:
    A `tff.templates.IterativeProcess` instance.
  """
  client_lr_callback = callbacks.create_reduce_lr_on_plateau(
      learning_rate=FLAGS.client_learning_rate,
      decay_factor=FLAGS.client_decay_factor,
      min_delta=FLAGS.min_delta,
      min_lr=FLAGS.min_lr,
      window_size=FLAGS.window_size,
      patience=FLAGS.patience)

  server_lr_callback = callbacks.create_reduce_lr_on_plateau(
      learning_rate=FLAGS.server_learning_rate,
      decay_factor=FLAGS.server_decay_factor,
      min_delta=FLAGS.min_delta,
      min_lr=FLAGS.min_lr,
      window_size=FLAGS.window_size,
      patience=FLAGS.patience)

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  def tff_model_fn():
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  return adaptive_fed_avg.build_fed_avg_process(
      tff_model_fn,
      client_lr_callback,
      server_lr_callback,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weight_fn=client_weight_fn)
