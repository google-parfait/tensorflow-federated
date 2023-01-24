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

from collections.abc import Callable, Sequence
from typing import Optional, TypeVar

import numpy as np

#  Settings for a multiplicative linear congruential generator (aka Lehmer
#  generator) suggested in 'Random Number Generators: Good
#  Ones are Hard to Find' by Park and Miller.
MLCG_MODULUS = 2 ** (31) - 1
MLCG_MULTIPLIER = 16807

# Type variable for matching the input and output of `build_sampling_fn`
T = TypeVar('T')


def build_uniform_sampling_fn(
    sample_range: Sequence[T],
    replace: bool = False,
    random_seed: Optional[int] = None,
) -> Callable[[int, int], list[T]]:
  """Builds the function for sampling from the input iterator at each round.

  If an integer `random_seed` is provided, we set a random seed before sampling
  clients according to a multiplicative linear congruential generator (aka
  Lehmer generator, see 'The Art of Computer Programming, Vol. 3' by Donald
  Knuth for reference). This does not affect model initialization, shuffling, or
  other such aspects of the federated training process.

  Args:
    sample_range: A 1-D array-like sequence, to be used as input to
      `np.random.choice`. Samples are generated randomly from the elements of
      the sequence.
    replace: A boolean indicating whether the sampling is done with replacement
      (True) or without replacement (False).
    random_seed: If an integer, it is used as a random seed for the client
      sampling process. If None, a nondeterministic seed is used.

  Returns:
    A function that takes as input an integer `round_num` and integer `size` and
    returns a list of `size` elements sampled (pseudo-)randomly from the input
    `sample_range`.
  """
  if isinstance(random_seed, int):
    mlcg_start = np.random.RandomState(random_seed).randint(1, MLCG_MODULUS - 1)

    def get_pseudo_random_int(round_num):
      return (
          pow(MLCG_MULTIPLIER, round_num, MLCG_MODULUS)
          * mlcg_start
          % MLCG_MODULUS
      )

  def sample_fn(round_num: int, size: int):
    if isinstance(random_seed, int):
      random_state = np.random.RandomState(get_pseudo_random_int(round_num))
    else:
      random_state = np.random.RandomState()
    try:
      return random_state.choice(
          sample_range, size=size, replace=replace
      ).tolist()
    except ValueError as e:
      raise ValueError(
          f'Failed to sample {size} clients from population of '
          f'size {len(sample_range)}.'
      ) from e

  return sample_fn
