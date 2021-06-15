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
"""Classes for creating and running baseline learning tasks."""

from typing import Any, Callable, List, Optional, Union

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.datasets import client_data

Loss = Union[tf.keras.losses.Loss, List[tf.keras.losses.Loss]]
CentralOrClientData = Union[tf.data.Dataset, client_data.ClientData]
PreprocessFnType = Union[Callable[[tf.data.Dataset], tf.data.Dataset],
                         computation_base.Computation]
PrintFn = Callable[[str], Any]
ModelSummaryFn = Callable[[PrintFn], None]


def _print_keras_model_summary(
    keras_model: tf.keras.Model,
    loss: Loss,
    loss_weights: Optional[List[float]] = None,
    metrics: Optional[List[tf.keras.metrics.Metric]] = None,
    print_fn: Optional[PrintFn] = print):
  """Prints a summary of a `tf.keras` model, loss, and metrics."""
  keras_model.summary(print_fn=print_fn)
  if isinstance(loss, tf.keras.losses.Loss):
    loss_summary = 'Loss: {}'.format(loss.name)
  else:
    loss_summary = 'Losses: {}'.format([x.name for x in loss])
  print_fn(loss_summary)

  if loss_weights is not None:
    loss_weights_summary = 'Loss Weights: {}'.format(loss_weights)
    print_fn(loss_weights_summary)

  if metrics is not None:
    metrics_summary = 'Metrics: {}'.format([x.name for x in metrics])
    print_fn(metrics_summary)


class BaselineTask(object):
  """A class containing data and model information for running a learning task.

  Attributes:
    train_data: A `tff.simulation.datasets.ClientData` for training.
    test_data: The test data for the baseline task. Can be a
      `tff.simulation.datasets.ClientData` or a `tf.data.Dataset`.
    validation_data: The validation data for the baseline task. Can be one of
      `tff.simulation.datasets.ClientData`, `tf.data.Dataset`, or `None` if the
      task does not have a validation dataset.
    train_preprocess_fn: A callable mapping accepting and return
      `tf.data.Dataset` instances, used for preprocessing train datasets. Set to
      `None` if no train preprocessing occurs for the task.
    eval_preprocess_fn: A callable mapping accepting and return
      `tf.data.Dataset` instances, used for preprocessing evaluation datasets.
      Set to `None` if no eval preprocessing occurs for the task.
    model_builder: A no-arg callable returning a `tff.learning.Model` for the
      learning task.
    element_type_structure: A nested structure of `tf.TensorSpec` objects
      defining the type of the elements contained in datasets associated to this
      task. This also matches the expected input structure of the result of
      `model_builder`.
  """

  def __init__(self,
               task_datasets: task_data.BaselineTaskDatasets,
               model_builder: Callable[[], model.Model],
               model_summary_fn: Optional[ModelSummaryFn] = None):
    """Creates a `BaselineTask`.

    Args:
      task_datasets: A `tff.simulation.baselines.BaselineTaskDatasets`.
      model_builder: A no-arg callable returning a `tff.learning.Model`.
      model_summary_fn: An optional callable that prints a summary of the model
        being used in the task. If `None`, calling
        `BaselineTask.get_model_summary` on the resulting task will print out a
        summary of the model's input spec.

    Raises:
      ValueError: If the element type structure of `task_datasets` does not
        match the input spec of `model_builder()`.
    """
    self._task_datasets = task_datasets
    self._model_builder = model_builder
    placeholder_model = model_builder()
    if placeholder_model.input_spec != task_datasets.element_type_structure:
      raise ValueError('The element type structure of task_datasets must match '
                       'the input spec of the tff.learning.Model provided.')

    if model_summary_fn is None:

      def get_model_summary(print_fn: PrintFn = print):
        print_fn('Model input spec: {}'.format(
            task_datasets.element_type_structure))

      self._model_summary_fn = get_model_summary
    else:
      self._model_summary_fn = model_summary_fn

  @property
  def train_data(self) -> client_data.ClientData:
    return self._task_datasets.train_data

  @property
  def test_data(self) -> CentralOrClientData:
    return self._task_datasets.test_data

  @property
  def validation_data(self) -> Optional[CentralOrClientData]:
    return self._task_datasets.validation_data

  @property
  def train_preprocess_fn(self) -> Optional[PreprocessFnType]:
    return self._task_datasets.train_preprocess_fn

  @property
  def eval_preprocess_fn(self) -> Optional[PreprocessFnType]:
    return self._task_datasets.eval_preprocess_fn

  @property
  def model_builder(self) -> Callable[[], model.Model]:
    return self._model_builder

  @property
  def element_type_structure(self):
    return self._element_type_structure

  def sample_train_clients(
      self,
      num_clients: int,
      replace: bool = False,
      random_seed: Optional[int] = None) -> List[tf.data.Dataset]:
    """Samples training clients uniformly at random.

    Args:
      num_clients: A positive integer representing number of clients to be
        sampled.
      replace: Whether to sample with replacement. If set to `False`, then
        `num_clients` cannot exceed the number of training clients in the
        associated train data.
      random_seed: An optional integer used to set a random seed for sampling.
        If no random seed is passed or the random seed is set to `None`, this
        will attempt to set the random seed according to the current system time
        (see `numpy.random.RandomState` for details).

    Returns:
      A list of `tf.data.Dataset` instances representing the client datasets.
    """
    return self._task_datasets.sample_train_clients(
        num_clients, replace=replace, random_seed=random_seed)

  def get_centralized_test_data(self) -> tf.data.Dataset:
    """Returns a `tf.data.Dataset` of test data for the task.

    If the baseline task has centralized data, then this method will return
    the centralized data after applying preprocessing. If the test data is
    federated, then this method will first amalgamate the client datasets into
    a single dataset, then apply preprocessing.
    """
    return self._task_datasets.get_centralized_test_data()

  def dataset_summary(self, print_fn: PrintFn = print):
    """Prints a summary of the task datasets.

    To capture the summary, you can use a custom print function. For example,
    setting `print_fn = summary_list.append` will cause each of the lines above
    to be appended to `summary_list`.

    For more details on the output of this method, see
    `tff.simulation.baselines.BaselineTaskDatasets.summary`.

    Args:
      print_fn: An optional callable accepting string inputs. Used to print each
        row of the summary. Defaults to `print` if not specified.
    """
    self._task_datasets.summary(print_fn=print_fn)

  def model_summary(self, print_fn: PrintFn = print):
    """Prints a summary of the model used for the learning task.

    To capture the summary, you can use a custom print function. For example,
    setting `print_fn = summary_list.append` will cause each of the lines above
    to be appended to `summary_list`.

    Args:
      print_fn: An optional callable accepting string inputs. Used to print each
        row of the summary. Defaults to `print` if not specified.
    """
    self._model_summary_fn(print_fn)

  @classmethod
  def from_data_and_keras_model(
      cls,
      task_datasets: task_data.BaselineTaskDatasets,
      keras_model: tf.keras.Model,
      loss: Loss,
      loss_weights: Optional[List[float]] = None,
      metrics: Optional[List[tf.keras.metrics.Metric]] = None):
    """Creates a baseline learning task from datasets and a `tf.keras` model.

    Args:
      task_datasets: A `tff.simulation.baselines.BaselineTaskDatasets`.
      keras_model: A `tf.keras.Model` object that is not compiled. The input
        layer of the model must be compatible with inputs of type matching
        `task_datasets.element_type_structure`.
      loss: A single `tf.keras.losses.Loss` or a list of losses-per-output. For
        more information, see `tff.learning.from_keras_model`.
      loss_weights: (Optional) A list of Python floats used to weight the loss
        contribution of each model output (when providing a list of losses for
        the `loss` argument).
      metrics: (Optional) a list of `tf.keras.metrics.Metric` objects.

    Returns:
      A `tff.simulation.baselines.BaselineTask`.
    """

    def model_builder() -> model.Model:
      return keras_utils.from_keras_model(
          keras_model=keras_model,
          loss=loss,
          input_spec=task_datasets.element_type_structure,
          loss_weights=loss_weights,
          metrics=metrics)

    def model_summary_fn(print_fn: PrintFn = print):
      _print_keras_model_summary(
          keras_model=keras_model,
          loss=loss,
          loss_weights=loss_weights,
          metrics=metrics,
          print_fn=print_fn)

    baseline_task = cls(
        task_datasets, model_builder, model_summary_fn=model_summary_fn)

    return baseline_task
