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
"""Library for building scheduled FedAvg iterative processes from flags.

Here, scheduled iterative processes incorporate client and server learning
rate schedules directly. For more details on the learning rate scheduling
functions, see optimizer_utils.py.
"""

from typing import Callable, List, Optional

from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shared import fed_avg_schedule
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.utils import utils_impl

# Defining optimizer flags
with utils_impl.record_hparam_flags():
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_lr_schedule_flags('client')
  optimizer_utils.define_lr_schedule_flags('server')

FLAGS = flags.FLAGS

# Convenience type aliases.
ModelBuilder = Callable[[], tff.learning.Model]
LossBuilder = Callable[[], tf.keras.losses.Loss]
MetricsBuilder = Callable[[], List[tf.keras.metrics.Metric]]
ClientWeightFn = Callable[..., float]


def from_flags(
    input_spec,
    model_builder: ModelBuilder,
    loss_builder: LossBuilder,
    metrics_builder: MetricsBuilder,
    client_weight_fn: Optional[ClientWeightFn] = None,
) -> tff.templates.IterativeProcess:
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
    A `tff.templates.IterativeProcess`.
  """
  # TODO(b/147808007): Assert that model_builder() returns an uncompiled keras
  # model.
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  client_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('client')
  server_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('server')

  model_input_spec = input_spec

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=model_input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  return fed_avg_schedule.build_fed_avg_process(
      model_fn=tff_model_fn,
      client_optimizer_fn=client_optimizer_fn,
      client_lr=client_lr_schedule,
      server_optimizer_fn=server_optimizer_fn,
      server_lr=server_lr_schedule,
      client_weight_fn=client_weight_fn)
