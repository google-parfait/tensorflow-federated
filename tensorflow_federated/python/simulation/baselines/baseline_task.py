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
"""Utilities for baseline tasks."""

from collections.abc import Callable

import attrs
import tensorflow as tf

from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.simulation.baselines import task_data


@attrs.define(frozen=True, init=True)
class BaselineTask:
  """Specification for a baseline learning simulation.

  Attributes:
    datasets: A `tff.simulation.baselines.BaselineTaskDatasets` object
      specifying dataset-related aspects of the task, including training data
      and preprocessing functions.
    model_fn: A no-arg callable returning a `tff.learning.models.VariableModel`
      used for the task. Note that `model_fn().input_spec` must match
      `datasets.element_type_structure`.
  """

  datasets: task_data.BaselineTaskDatasets = attrs.field(
      validator=attrs.validators.instance_of(task_data.BaselineTaskDatasets)
  )
  model_fn: Callable[[], variable.VariableModel] = attrs.field(
      validator=attrs.validators.is_callable()
  )

  def __attrs_post_init__(self):
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    with tf.Graph().as_default():
      tff_model = self.model_fn()
    if not isinstance(tff_model, variable.VariableModel):
      raise TypeError(
          'Expected model_fn to output a tff.learning.models.VariableModel, '
          'found {} instead'.format(type(tff_model))
      )

    dataset_element_spec = self.datasets.element_type_structure
    model_input_spec = tff_model.input_spec

    if dataset_element_spec != model_input_spec:
      raise ValueError(
          'Dataset element spec and model input spec do not match.'
          'Found dataset element spec {}, but model input spec {}'.format(
              dataset_element_spec, model_input_spec
          )
      )
