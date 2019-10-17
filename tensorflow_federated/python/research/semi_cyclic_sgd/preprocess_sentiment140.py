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
"""Preprocesses Sentiment140 data for use in training.

Does stuff like some string replacements, converting dates, shuffling, splitting
into train+test data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import random
import re
import sys

from absl import app

INPUT = '/tmp/sc_paper/raw_data/training.1600000.processed.noemoticon.csv'
TRAIN_OUTPUT = '/tmp/sc_paper/raw_data/train.csv'
TEST_OUTPUT = '/tmp/sc_paper/raw_data/test.csv'
DICT_OUTPUT = '/tmp/sc_paper/raw_data/dict.txt'
TRAIN_SPLIT = 0.9


def replace_usernames(text):
  # Didn't find a good regexp for usernames so kept them for now as is.
  return text


def replace_uris(text):
  # Replaces http://abcd and https://abcd with URI abcd.
  # I'd like to replace the whole URI, but matching URIs with regexps is tricky;
  # not sure how to determine the end of a URI in twitter reliably. So I just
  # replace the scheme with "URI", and leave the rest; in all likelihood the
  # rest won't end up in the dictionary and is ignored, and if not, maybe it
  # contains useful info.
  return text.replace('http://', 'URI ').replace('https://', 'URI ')  # pytype: disable=attribute-error


def replace_repeated_characters(text):
  """Replaces every 3+ repetition of a character with a 2-repetition."""
  if not text:
    return text
  res = text[0]
  c_prev = text[0]
  n_reps = 0
  for c in text[1:]:
    if c == c_prev:
      n_reps += 1
      if n_reps < 2:
        res += c
    else:
      n_reps = 0
      res += c
    c_prev = c
  return res


def split_line(text):
  # Add @? Right now that gets stripped out.
  return re.findall(r"[\w']+|[.,!?;]", text)


def main(unused_args):
  # There's some unicode errors in the input, ignore them (also Py2.7 behavior,
  # which was used for the paper).
  with open(INPUT, 'r', errors='ignore') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    lines = []
    print('reading CSV file')
    unigrams = {}
    i = 0
    for row in csv_reader:
      if row[0] == '0':
        label = 0
      elif row[0] == '4':
        label = 1
      else:
        raise ValueError('Invalid label: {}'.format(row[0]))
      row[0] = label
      text = row[5]
      text = replace_usernames(text)
      text = replace_uris(text)
      text = replace_repeated_characters(text)
      row[5] = text
      lines.append(row)
      for w in split_line(text):
        if w in unigrams:
          unigrams[w] = unigrams[w] + 1
        else:
          unigrams[w] = 1
      i = i + 1
      if i % 100000 == 0:
        print('read {} rows'.format(i))

    # lines are sorted by sentiment, then by date. Shuffle.
    print('Shuffling data')
    random.shuffle(lines)

    # Split into train and test data.
    split_index = int(len(lines) * TRAIN_SPLIT)
    print('Writing training data')
    with open(TRAIN_OUTPUT, 'w') as f:
      w = csv.writer(f)
      for l in lines[0:split_index]:
        w.writerow(l)
    print('Writing test data')
    with open(TEST_OUTPUT, 'w') as f:
      w = csv.writer(f)
      for l in lines[split_index:]:
        w.writerow(l)

    # Write dictionary.
    unigrams_sorted = sorted(
        list(unigrams.items()), key=lambda kv: kv[1], reverse=True)
    print(unigrams_sorted[0:20])
    print('{} lines read'.format(len(lines)))
    with open(DICT_OUTPUT, 'w') as f:
      for w in unigrams_sorted:
        f.write(w[0] + '\n')


if __name__ == '__main__':
  if not sys.version_info >= (3, 0):
    print('This script requires Python3 to run.')
    sys.exit(1)
  app.run(main)
