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
"""Classes for loading and preprocessing data for federated baseline tasks."""

import collections
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.simulation.datasets import client_data

CentralOrClientData = Union[tf.data.Dataset, client_data.ClientData]
PreprocessFnType = Union[
    Callable[[tf.data.Dataset], tf.data.Dataset], computation_base.Computation
]


def _get_element_spec(
    data: CentralOrClientData, preprocess_fn: Optional[PreprocessFnType] = None
):
  """Determines the element type of a dataset after preprocessing."""
  if isinstance(data, client_data.ClientData):
    if preprocess_fn is not None:
      preprocessed_data = data.preprocess(preprocess_fn)
    else:
      preprocessed_data = data
    element_spec = preprocessed_data.element_type_structure
  else:
    if preprocess_fn is not None:
      preprocessed_data = preprocess_fn(data)
    else:
      preprocessed_data = data
    element_spec = preprocessed_data.element_spec
  return element_spec


class BaselineTaskDatasets:
  """A convenience class for a task's data and preprocessing logic.

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
    element_type_structure: A nested structure of `tf.TensorSpec` objects
      defining the type of the elements contained in datasets associated to this
      task.
  """

  def __init__(
      self,
      train_data: client_data.ClientData,
      test_data: CentralOrClientData,
      validation_data: Optional[CentralOrClientData] = None,
      train_preprocess_fn: Optional[PreprocessFnType] = None,
      eval_preprocess_fn: Optional[PreprocessFnType] = None,
  ):
    """Creates a `BaselineTaskDatasets`.

    Args:
      train_data: A `tff.simulation.datasets.ClientData` for training.
      test_data: A `tff.simulation.datasets.ClientData` or a `tf.data.Dataset`
        for computing test metrics.
      validation_data: An optional `tff.simulation.datasets.ClientData` or a
        `tf.data.Dataset` for computing validation metrics.
      train_preprocess_fn: An optional callable accepting and returning a
        `tf.data.Dataset`, used to perform dataset preprocessing for training.
        If set to `None`, we use the identity map for all train preprocessing.
      eval_preprocess_fn: An optional callable accepting and returning a
        `tf.data.Dataset`, used to perform evaluation (eg. validation, testing)
        preprocessing. If `None`, evaluation preprocessing will be done via the
        identity map.

    Raises:
      ValueError: If `train_data` and `test_data` have different element types
        after preprocessing with `train_preprocess_fn` and `eval_preprocess_fn`,
        or if `validation_data` is not `None` and has a different element type
        than the test data.
    """
    self._train_data = train_data
    self._test_data = test_data
    self._validation_data = validation_data
    self._train_preprocess_fn = train_preprocess_fn
    self._eval_preprocess_fn = eval_preprocess_fn

    if train_preprocess_fn is not None and not callable(train_preprocess_fn):
      raise ValueError('The train_preprocess_fn must be None or callable.')
    self._train_preprocess_fn = train_preprocess_fn

    if (eval_preprocess_fn is not None) and (not callable(eval_preprocess_fn)):
      raise ValueError('The eval_preprocess_fn must be None or callable.')
    self._eval_preprocess_fn = eval_preprocess_fn

    post_preprocess_train_type = _get_element_spec(
        train_data, train_preprocess_fn
    )
    if train_preprocess_fn is None:
      self._preprocess_train_data = train_data
    else:
      self._preprocess_train_data = train_data.preprocess(train_preprocess_fn)

    # TODO: b/249818282 - Add validation to ensure that this is compatible with
    # the post-processed test dataset.
    self._element_type_structure = post_preprocess_train_type

    if validation_data is not None:
      test_type = _get_element_spec(test_data)
      validation_type = _get_element_spec(validation_data)
      if test_type != validation_type:
        raise ValueError(
            'The validation set must be None, or have the same element type '
            'structure as the test data. Found test type {} and validation type'
            ' {}'.format(test_type, validation_type)
        )

    self._data_info = None

  @property
  def train_data(self) -> client_data.ClientData:
    return self._train_data

  @property
  def test_data(self) -> CentralOrClientData:
    return self._test_data

  @property
  def validation_data(self) -> Optional[CentralOrClientData]:
    return self._validation_data

  @property
  def train_preprocess_fn(self) -> Optional[PreprocessFnType]:
    return self._train_preprocess_fn

  @property
  def eval_preprocess_fn(self) -> Optional[PreprocessFnType]:
    return self._eval_preprocess_fn

  @property
  def element_type_structure(self):
    return self._element_type_structure

  def _record_dataset_information(self):
    """Records a summary of the train, test, and validation data."""
    data_info = collections.OrderedDict()
    data_info['header'] = ['Split', 'Dataset Type', 'Number of Clients']

    num_train_clients = len(self._train_data.client_ids)
    data_info['train'] = ['Train', 'Federated', num_train_clients]

    if isinstance(self._test_data, client_data.ClientData):
      test_type = 'Federated'
      num_test_clients = len(self._test_data.client_ids)
    else:
      test_type = 'Centralized'
      num_test_clients = 'N/A'
    data_info['test'] = ['Test', test_type, num_test_clients]

    if self._validation_data is not None:
      if isinstance(self._validation_data, client_data.ClientData):
        validation_type = 'Federated'
        num_validation_clients = len(self._validation_data.client_ids)
      else:
        validation_type = 'Centralized'
        num_validation_clients = 'N/A'
      data_info['validation'] = [
          'Validation',
          validation_type,
          num_validation_clients,
      ]

    return data_info

  def sample_train_clients(
      self,
      num_clients: int,
      replace: bool = False,
      random_seed: Optional[int] = None,
  ) -> list[tf.data.Dataset]:
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
    random_state = np.random.RandomState(seed=random_seed)
    client_ids = random_state.choice(
        self._preprocess_train_data.client_ids,
        size=num_clients,
        replace=replace,
    )
    return [
        self._preprocess_train_data.create_tf_dataset_for_client(x)
        for x in client_ids
    ]

  def get_centralized_test_data(self) -> tf.data.Dataset:
    """Returns a `tf.data.Dataset` of test data for the task.

    If the baseline task has centralized data, then this method will return
    the centralized data after applying preprocessing. If the test data is
    federated, then this method will first amalgamate the client datasets into
    a single dataset, then apply preprocessing.
    """
    test_data = self._test_data
    if isinstance(test_data, client_data.ClientData):
      test_data = test_data.create_tf_dataset_from_all_clients()
    preprocess_fn = self._eval_preprocess_fn
    if preprocess_fn is not None:
      test_data = preprocess_fn(test_data)
    return test_data

  def summary(self, print_fn: Callable[[str], Any] = print):
    """Prints a summary of the train, test, and validation data.

    The summary will be printed as a table containing information on the type
    of train, test, and validation data (ie. federated or centralized) and the
    number of clients each data structure has (if it is federated). For example,
    if the train data has 10 clients, and both the test and validation data are
    centralized, then this will print the following table:

    ```
    Split      |Dataset Type |Number of Clients |
    =============================================
    Train      |Federated    |10                |
    Test       |Centralized  |N/A               |
    Validation |Centralized  |N/A               |
    _____________________________________________
    ```

    In addition, this will print two lines after the table indicating whether
    train and eval preprocessing functions were passed in. In the example above,
    if we passed in a train preprocessing function but no eval preprocessing
    function, it would also print the lines:
    ```
    Train Preprocess Function: True
    Eval Preprocess Function: False
    ```

    To capture the summary, you can use a custom print function. For example,
    setting `print_fn = summary_list.append` will cause each of the lines above
    to be appended to `summary_list`.

    Args:
      print_fn: An optional callable accepting string inputs. Used to print each
        row of the summary. Defaults to `print` if not specified.
    """
    if self._data_info is None:
      self._data_info = self._record_dataset_information()
    data_info = self._data_info
    num_cols = len(data_info['header'])
    max_lengths = [0 for _ in range(num_cols)]
    for col_values in data_info.values():
      for j, col_value in enumerate(col_values):
        max_lengths[j] = max([len(str(col_value)), max_lengths[j]])

    col_lengths = [a + 1 for a in max_lengths]

    row_strings = []
    for col_values in data_info.values():
      row_string = ''
      for col_val, col_len in zip(col_values, col_lengths):
        row_string += '{col_val:<{col_len}}|'.format(
            col_val=col_val, col_len=col_len
        )
      row_strings.append(row_string)

    total_width = sum(col_lengths) + num_cols
    row_strings.insert(1, '=' * total_width)
    row_strings.append('_' * total_width)

    for x in row_strings:
      print_fn(x)

    train_preprocess_fn_exists = self._train_preprocess_fn is not None
    print_fn('Train Preprocess Function: {}'.format(train_preprocess_fn_exists))

    eval_preprocess_fn_exists = self._eval_preprocess_fn is not None
    print_fn('Eval Preprocess Function: {}'.format(eval_preprocess_fn_exists))
