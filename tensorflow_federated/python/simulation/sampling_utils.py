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
"""Utilities for sampling clients, either randomly or pseudo-randomly."""

import functools
from typing import Callable, List, Optional, Sequence, TypeVar

import numpy as np

from tensorflow_federated.python.simulation import client_data


#  Settings for a multiplicative linear congruential generator (aka Lehmer
#  generator) suggested in 'Random Number Generators: Good
#  Ones are Hard to Find' by Park and Miller.
MLCG_MODULUS = 2**(31) - 1
MLCG_MULTIPLIER = 16807


# Type variable for matching the input and output of `build_sampling_fn`
T = TypeVar('T')


def build_uniform_sampling_fn(
    sample_range: Sequence[T],
    size: int,
    replace: bool = False,
    random_seed: Optional[int] = None) -> Callable[[int], List[T]]:
  """Builds the function for sampling from the input iterator at each round.

  Args:
    sample_range: A 1-D array-like sequence, to be used as input to
    `np.random.choice`. Samples are generated randomly from the elements of the
    sequence.
    size: The number of samples to return each round.
    replace: A boolean indicating whether the sampling is done with replacement
      (True) or without replacement (False).
    random_seed: If random_seed is set as an integer, then we use it as a random
      seed for which clients are sampled at each round. In this case, we set a
      random seed before sampling clients according to a multiplicative linear
      congruential generator (aka Lehmer generator, see 'The Art of Computer
      Programming, Vol. 3' by Donald Knuth for reference). This does not affect
      model initialization, shuffling, or other such aspects of the federated
      training process.

  Returns:
    A function that takes as input an integer `round_num` and returns a list of
    elements sampled (pseudo-)randomly from the input `sample_range`.
  """
  if isinstance(random_seed, int):
    mlcg_start = np.random.RandomState(random_seed).randint(1, MLCG_MODULUS - 1)

    def get_pseudo_random_int(round_num):
      return pow(MLCG_MULTIPLIER, round_num,
                 MLCG_MODULUS) * mlcg_start % MLCG_MODULUS

  def sample(round_num, random_seed):
    if isinstance(random_seed, int):
      random_state = np.random.RandomState(get_pseudo_random_int(round_num))
    else:
      random_state = np.random.RandomState()
    return random_state.choice(
        sample_range, size=size, replace=replace).tolist()

  return functools.partial(sample, random_seed=random_seed)


def build_uniform_client_sampling_fn(
    dataset: client_data.ClientData,
    clients_per_round: int,
    random_seed: Optional[int] = None) -> Callable[[int], List[str]]:
  """Builds a function that (pseudo-)randomly samples clients.

  The function uniformly samples a number of clients (without replacement within
  a given round, but with replacement across rounds) and returns their ids.

  Args:
    dataset: A `tff.simulation.ClientData` object.
    clients_per_round: The number of client participants in each round.
    random_seed: If random_seed is set as an integer, then we use it as a random
      seed for which clients are sampled at each round. In this case, we set a
      random seed before sampling clients according to a multiplicative linear
      congruential generator (aka Lehmer generator, see 'The Art of Computer
      Programming, Vol. 3' by Donald Knuth for reference). This does not affect
      model initialization, shuffling, or other such aspects of the federated
      training process. Note that this will alter the global numpy random seed.

  Returns:
    A function that takes as input an integer `round_num` and returns a a list
    of ids corresponding to the uniformly sampled clients.
  """
  return build_uniform_sampling_fn(
      dataset.client_ids,
      size=clients_per_round,
      replace=False,
      random_seed=random_seed)
