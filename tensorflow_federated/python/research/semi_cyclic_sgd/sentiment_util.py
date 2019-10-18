# Lint as: python3
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
"""Tools to process Sentiment140 data (building a dict, tokenizing etc.)"""

import re


def line_to_word_ids(line, vocab):
  # Splits a line into words and maps them to word IDs from vocab.
  # Split given line/phrase into list of words
  words = re.findall(r"[\w']+|[.,!?;]", line)
  # Return IDs for known words.
  return [vocab[w] for w in words if w in vocab]


def bag_of_words(x_batch, bags, limit=0):
  # Converts a python list-of-list (batch x num_word_ids) to a numpy
  # matrix (batch_size x vocab_size), each element indicating
  # the number of occurrences of a word in a sentence (capped by "limit").
  for i, word_ids in enumerate(x_batch):
    for word_id in word_ids:
      if limit == 0 or bags[i, word_id] < limit:
        bags[i, word_id] += 1


def val_to_vec(size, val):
  # Creates a one-hot vector, length size, 1 at index val.
  assert 0 <= val < size
  vec = [0] * size
  vec[int(val)] = 1
  return vec
