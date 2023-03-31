# Copyright 2023, The TensorFlow Federated Authors.
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
"""Subsampling and tuning interface for subsampled SecAggIBLT.

The class interface implements the subsampling function needed for subsampled
SecAggIBLT and the tuning function for adjusting the subsampling parameter based
on server measurements. A common use is to combine it with an IBLT aggregation
process and apply the subsampling function to users' local datasets before
passing them to the aggregation process. The subsampling parameter can be
updated at each round to obtain instance-dependent algorithm.
"""

import abc
import collections

import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory


class SubsampleProcess(abc.ABC):
  """Describes the interface for subsampling tuning.

  The class abstracts methods needed for subsampling users' local datasets
  before aggregating them using the IBLT factory and the correponding tuning
  method based on server measurements.

  The class should provide the following methods needed in a (adaptive)
  subsampled IBLT process.

  1. Intialization of the subsampling parameter.
  2. Subsampling function performed on each user's local dataset.
  3. Check for whether the process is an adptive process.
  4. Tuning method for subsampling parameter based on the current subsampling
  parameter and the server measurements in the current round.

  The functions will be typically used within a `IterativeProcess` together with
  other federated computations. And the update function will be usually use the
  returned measurements as an input. Hence seperating the subsample function and
  the update function would offer more flexibility in their implementations.
  Below is an example using ThresholdSamplingProcess over multiple rounds.

  In the example, we assume each call of get_client_datasets(...) returns a list
  of client local datasets. We also assume iblt_aggregation is properly
  intialized and the next() returns measurements of the SecAggIBLT process.

  ```
  threshold_sampling = ThresholdSamplingProcess(init_param = 10.0)
  sampling_param = threshold_sampling.get_init_param()
  for round in range(num_rounds):
    client_datasets = get_client_datasets(...)
    subsampled_datasets = []
    for dataset in client_datasets:
      subsampled_dataset = threshold_sampling.subsample_fn(dataset,
                                                            sampling_param)
      subsampled_datasets.append(subsampled_dataset)
    measurements, _ = iblt_aggregation.next(..., subsampled_datasets)
    if threshold_sampling.is_adaptive():
      sampling_param = threshold_sampling.update(sampling_param, measurements)
  ```
  """

  @abc.abstractmethod
  def is_adaptive(self) -> bool:
    """Return whether the process is adaptive."""

  @abc.abstractmethod
  def update(
      self, subsampling_param_old: float, measurements: tf.Tensor
  ) -> float:
    """Update the sample parameter based on the return of an IBLT round.

    Args:
      subsampling_param_old: subsampling parameter for the current round.
      measurements: a tf.Tensor containing the returned measurements of an IBLT
        aggregation process in the current round.

    Returns:
      The subsampling parameter to use in the next round.
    """
    raise NotImplementedError('Not an adaptive process!')

  @abc.abstractmethod
  def get_init_param(self):
    """Returns the initial subsampling parameter."""

  @abc.abstractmethod
  def subsample_fn(
      self, client_data: tf.data.Dataset, subsampling_param: float
  ):
    """Performs subsampling at clients.

    Args:
      client_data: a `tf.data.Dataset` that represents client dataset.
      subsampling_param: subsampling parameter to use.

    Returns:
      subsampled client dataset with the same format as client_data.
    """


class ThresholdSamplingProcess(SubsampleProcess):
  """Implements threshold sampling without tuning.

  Threshold sampling performs the following randomization of the user's local
  histogram. Let h(x) be the count of element x at user's local dataset.
  Given a threshold t, subsample_fn returns h' satisfying:

  ```
  if h(x) > t: return h(x)
  else if h(x)/t > random(): return t
  else: return 0
  ```

  The current implementation only supports histograms with nonnegative values.

  Example: if user's local histogram is {'a': 3, 'b': 5, 'c': 1}. After sampling
  with threshold 4. 'b' will always be in the subsampled dataset with count 5.
  'a' will appear with probability 3/4 and 'c' will appear with probability 1/4.
  The count of 'a' and 'c' will always be 4 if they appear.

  See more details at:
  https://nickduffield.net/download/papers/DLT05-optimal.pdf
  """

  def __init__(self, init_param: float):
    """See base class. Raise ValueError if init_param is less than 1."""
    if init_param < 1:
      raise ValueError('Threshold must be at least 1!')
    self._init_param = init_param

  def is_adaptive(self) -> bool:
    """Return whether the process is adaptive."""
    return False

  def update(
      self, subsampling_param_old: float, measurements: tf.Tensor
  ) -> float:
    """See base class."""
    raise NotImplementedError('Not an adaptive process!')

  def get_init_param(self):
    """Returns the initial subsampling parameter."""
    return self._init_param

  def subsample_fn(
      self, client_data: tf.data.Dataset, subsampling_param: float
  ):
    """See base class. Raise ValueError if client data has negative counts."""

    generator = tf.random.Generator.from_non_deterministic_state()

    @tf.function
    def threshold_sampling(element):
      count = element[iblt_factory.DATASET_VALUE]
      tf.debugging.assert_non_negative(
          count, 'Current implementation only supports positive values!'
      )
      if count >= subsampling_param:
        return element
      random_val = generator.uniform(
          shape=(), minval=0, maxval=subsampling_param, dtype=count.dtype
      )
      thresholded_val = subsampling_param if count > random_val else 0
      return collections.OrderedDict([
          (iblt_factory.DATASET_KEY, element[iblt_factory.DATASET_KEY]),
          (
              iblt_factory.DATASET_VALUE,
              tf.cast([thresholded_val], dtype=count.dtype),
          ),
      ])

    subsampled_client_data = client_data.map(threshold_sampling)
    return subsampled_client_data.filter(
        lambda x: x[iblt_factory.DATASET_VALUE][0] > 0
    )
