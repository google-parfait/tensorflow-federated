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
"""Script for training + evaluating models for Semi-Cyclic SGD paper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import datetime
import os
import random
import sys
import time

from absl import app
from absl import flags
import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.research.semi_cyclic_sgd import sentiment_util as su

FLAGS = flags.FLAGS
RAWDATA = '/tmp/sc_paper/raw_data/'
flags.DEFINE_integer('task_id', -1,
                     'Task id specifying which experiment to run.')
flags.DEFINE_string('log_file', '/tmp/sc.log', 'Log file path.')
flags.DEFINE_string('training_data', os.path.join(RAWDATA, 'train.csv'),
                    'Path to training data file.')
flags.DEFINE_string('test_data', os.path.join(RAWDATA, 'test.csv'),
                    'Path to test data file.')
flags.DEFINE_string(
    'dictionary', os.path.join(RAWDATA, 'dict.txt'),
    'Dictionary file (one word per line, descending by '
    'frequency).')
flags.DEFINE_float('lr', 0.215, 'Learning rate.')
flags.DEFINE_integer('batch_size', 128, 'Minibatch size.')
flags.DEFINE_integer('num_days', 10, 'Number of days to train')
flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size (0: unlimited).')
flags.DEFINE_integer('bow_limit', 0,
                     'Max num of words in bow presentation (0: unlimited).')
flags.DEFINE_integer(
    'test_after', 0, 'compute test data metrics every test_after training'
    ' examples. If 0, don\'t.')
flags.DEFINE_string(
    'mode', 'iid',
    'Train/test mode. One of iid, sep (train separate models), sc '
    '(semi-cyclic + pluralistic)')
flags.DEFINE_integer(
    'num_groups', 2, 'Number of groups the data will be split up into. '
    'Every day, num_groups blocks of data are created, each '
    'with num_train_examples_per_day/num_groups examples.')
flags.DEFINE_integer(
    'num_train_examples_per_day', 144000,
    'Number of examples per day for training. Default is picked such that '
    'for a 90/10 train/test split, ten days corresponds to '
    'one training pass over the complete dataset (which is sufficient '
    'for convergence).')
flags.DEFINE_float('bias', 0.0,
                   'Peak percentage of pos or neg examples to drop')
flags.DEFINE_integer(
    'replica', 0, 'Replica - this script may be invoked multiple times with '
    'identical parameters, to compute means+stds. This is the one '
    'flag that varies, and used as a seed')


class Model(object):
  """Class representing a logistic regression model using bag-of-words."""

  def __init__(self, lr, vocab, bow_limit):
    self.vocab = vocab
    self.input_dim = len(self.vocab)
    self.lr = lr
    self.num_classes = 2
    self.bow_limit = bow_limit
    self._optimizer = None

  @property
  def optimizer(self):
    """Optimizer to be used by the model."""
    if self._optimizer is None:
      self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    return self._optimizer

  def create_model(self):
    """Creates a TF model and returns ops necessary to run training/eval."""
    features = tf.placeholder(tf.float32, [None, self.input_dim])
    labels = tf.placeholder(tf.float32, [None, self.num_classes])

    w = tf.Variable(tf.random_normal(shape=[self.input_dim, self.num_classes]))
    b = tf.Variable(tf.random_normal(shape=[self.num_classes]))

    pred = tf.nn.softmax(tf.matmul(features, w) + b)

    loss = tf.reduce_mean(
        -tf.reduce_sum(labels * tf.log(pred), reduction_indices=1))
    train_op = self.optimizer.minimize(
        loss=loss, global_step=tf.train.get_or_create_global_step())

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    eval_metric_op = tf.count_nonzero(correct_pred)

    return features, labels, train_op, loss, eval_metric_op

  def process_x(self, raw_batch):
    x_batch = [e[0][4] for e in raw_batch]  # list of lines/phrases
    bags = np.zeros((len(raw_batch), len(self.vocab)))
    su.bag_of_words(x_batch, bags, self.bow_limit)
    return bags

  def process_y(self, raw_batch):
    y_batch = [int(e[1]) for e in raw_batch]
    y_batch = [su.val_to_vec(self.num_classes, e) for e in y_batch]
    y_batch = np.array(y_batch)
    return y_batch


class CyclicDataGenerator(object):
  """Generates minibatches from data grouped by day & group.

  The class does not handle loading or preprocessing (such as shuffling) of
  data, and subclasses should implement that in their constructor.
  """

  def __init__(self, logger, num_groups, num_examples_per_day_per_group,
               batch_size):
    # self.data has num_groups list. Each contains the entire data of that
    # group, which is processed in a round-robin manner, see comments on get().
    assert batch_size != 0
    self.logger = logger
    self.num_groups = num_groups
    self.data = [[] for _ in range(0, num_groups)]  # can't use [[]]*num_groups
    self.num_examples_per_day_per_group = num_examples_per_day_per_group
    self.batch_size = batch_size

  def get(self, day, group):
    """Gets data for training - a generator representing one block."""
    # Compute where inside the group to start. We assume round-robin processing,
    # e.g. for a group of size 10, 2 groups, 8 examples per day, we'd have 4
    # examples per day per group, so on day 1 we'd return examples 0..3, on day
    # 2 examples 4..7, on day 3 examples 8,9,0,1, etc. That also works when we
    # have to iterate through a group more than once per day.
    assert group < self.num_groups
    start_index = day * self.num_examples_per_day_per_group
    end_index = (day + 1) * self.num_examples_per_day_per_group

    for i in range(start_index, end_index, self.batch_size):
      group_size = len(self.data[group])
      assert group_size >= self.batch_size
      i_mod = i % group_size
      if i_mod + self.batch_size <= group_size:
        result = self.data[group][i_mod:i_mod + self.batch_size]
        if not result:
          print(day, group, self.num_groups, start_index, end_index, i,
                group_size, i_mod, self.batch_size)
      else:
        result = self.data[group][i_mod:]
        remainder = self.batch_size - len(result)
        result.extend(self.data[group][:remainder])
        if not result:
          print(day, group, self.num_groups, start_index, end_index, i,
                group_size, i_mod, self.batch_size)
      yield result

  def get_test_data(self, group):
    # Gets all data for a group, ignoring num_examples_per_day. Used for test
    # data where splitting into days may be undesired.
    assert group < self.num_groups
    for i in range(0,
                   len(self.data[group]) - self.batch_size + 1,
                   self.batch_size):
      yield self.data[group][i:i + self.batch_size]

  def process_row(self, row, vocab):
    """Utility function that preprocesses a Sentiment140 training example.

    Args:
      row: a row, split into items, from the Sentiment140 CSV file.
      vocab: a vocabulary for tokenization.  The example is processed by
        converting the label to an integer, and tokenizing the text of the post.
        The argument row is modified in place.
    """
    if row[0] == '1':
      row[0] = 1
    elif row[0] == '0':
      row[0] = 0
    else:
      raise ValueError('label neither 0 nor 1, but: type %s, value %s' %
                       (type(row[0]), str(row[0])))
    row[5] = su.line_to_word_ids(row[5], vocab)


class IidDataGenerator(CyclicDataGenerator):
  """Generates i.i.d.

  data by evenly distributing data across groups.

  Assumes
  data is already shuffled (though we shuffle another time).
  """

  def __init__(self,
               logger,
               path,
               vocab,
               num_groups,
               num_examples_per_day_per_group,
               batch_size=0):
    CyclicDataGenerator.__init__(self, logger, num_groups,
                                 num_examples_per_day_per_group, batch_size)
    with open(path, 'r') as f:
      csv_reader = csv.reader(f, delimiter=',')
      i = 0
      for row in csv_reader:
        self.process_row(row, vocab)
        label = row[0]
        self.data[i % self.num_groups].append([row[1:], label])
        i += 1
    for g in range(0, self.num_groups):
      random.shuffle(self.data[g])


class NonIidDataGenerator(CyclicDataGenerator):
  """A data generator for block cyclic data."""

  def __init__(self,
               logger,
               path,
               vocab,
               num_groups,
               num_examples_per_day_per_group,
               bias,
               batch_size=0):
    CyclicDataGenerator.__init__(self, logger, num_groups,
                                 num_examples_per_day_per_group, batch_size)
    # Bias parameter b=0..1 specifies how much to bias. In group 0, we drop
    # b*100% of the negative examples - let's call this biasing by +b.
    # In group num_groups/2+1, we drop b*100% of the positive
    # examples - say, biasing by -b. And then we go back to +b again, to have
    # some continuity. That interpolation only works cleanly with an even
    # num_groups.
    assert num_groups % 2 == 0
    # Python 2: type(x/2) == int, Python 3: type(x/2) == float.
    biases = np.interp(
        range(num_groups // 2 + 1), [0, num_groups / 2],
        [-bias, bias]).tolist()
    biases.extend(biases[-2:0:-1])
    with open(path, 'r') as f:
      csv_reader = csv.reader(f, delimiter=',')
      for row in csv_reader:
        self.process_row(row, vocab)
        label = row[0]
        t = datetime.datetime.strptime(row[2], '%a %b %d %H:%M:%S %Z %Y')
        # 1. Split by time of day into num_groups.
        assert 24 % num_groups == 0
        group = int(t.hour / (24 / num_groups))
        # 2. Introduce further bias by dropping examples.
        if bias > 0.0:
          r = random.random()
          b = biases[group]
          # Drop neg?
          if b < 0:
            if label == 1 or r >= abs(b):
              self.data[group].append([row[1:], label])
          else:
            if label == 0 or r >= b:
              self.data[group].append([row[1:], label])
        else:
          # No biasing, add unconditionally.
          self.data[group].append([row[1:], label])
    for g in range(0, self.num_groups):
      logger.log('group %d: %d examples' % (g, len(self.data[g])))
      random.shuffle(self.data[g])


class Logger(object):
  """A logger that logs to stdout and to a logfile, with throttling."""

  def __init__(self, interval=1):
    self.t = 0
    self.interval = interval
    self.out_file = open(FLAGS.log_file, 'w')

  def maybe_log(self, message):
    """Log if the last call to maybe_log was more than interval seconds ago."""
    cur_time = time.time()
    if self.t == 0 or self.t + self.interval < cur_time:
      print(message)
      print(message, file=self.out_file)
      self.t = cur_time

  def log(self, message):
    print(message)
    print(message, file=self.out_file)


def log_config(logger):
  """Logs the configuration of this run, so it can be used in the analysis phase."""
  logger.log('== Configuration ==')
  logger.log('task_id=%d' % FLAGS.task_id)
  logger.log('lr=%f' % FLAGS.lr)
  logger.log('vocab_size=%s' % FLAGS.vocab_size)
  logger.log('batch_size=%s' % FLAGS.batch_size)
  logger.log('bow_limit=%s' % FLAGS.bow_limit)
  logger.log('training_data=%s' % FLAGS.training_data)
  logger.log('test_data=%s' % FLAGS.test_data)
  logger.log('num_groups=%d' % FLAGS.num_groups)
  logger.log('num_days=%d' % FLAGS.num_days)
  logger.log('num_train_examples_per_day=%d' % FLAGS.num_train_examples_per_day)
  logger.log('mode=%s' % FLAGS.mode)
  logger.log('bias=%f' % FLAGS.bias)
  logger.log('replica=%d' % FLAGS.replica)


def test(test_data,
         model,
         sess,
         eval_metric_op,
         features,
         labels,
         logger,
         d,
         g,
         num_train_examples,
         prefix=''):
  """Tests the current model on all the data from test_data for group g."""
  cur_time = time.time()
  num_correct = 0
  num_test_examples = 0
  t_process_x = 0
  t_process_y = 0
  t_process_tf = 0
  for b in test_data.get_test_data(g):
    t1 = time.clock()
    x = model.process_x(b)
    t2 = time.clock()
    y = model.process_y(b)
    t3 = time.clock()
    num_test_examples = num_test_examples + len(x)
    num_correct_ = sess.run([eval_metric_op], {features: x, labels: y})
    t4 = time.clock()
    num_correct = num_correct + num_correct_[0]
    t_process_x = t_process_x + (t2 - t1)
    t_process_y = t_process_y + (t3 - t2)
    t_process_tf = t_process_tf + (t4 - t3)
  dt = time.time() - cur_time
  logger.log(
      '%sday %d, group %g: num_train_examples %d (dt=%ds): num correct: %d/%d (%f)'
      % (prefix, d, g, num_train_examples, dt, num_correct, num_test_examples,
         num_correct / float(num_test_examples)))


def init(logger):
  """Loads + groups data, dictionary."""
  vocab = {}
  i = 0
  with open(FLAGS.dictionary) as f:
    for l in f:
      w = l.strip()
      vocab[w] = i
      i = i + 1
      if FLAGS.vocab_size > 0 and i >= FLAGS.vocab_size:
        break
  logger.log('Read vocabulary with %d words' % len(vocab))
  logger.log('Loading training & testing data')
  if FLAGS.mode == 'iid':
    training_data = IidDataGenerator(
        logger, FLAGS.training_data, vocab, FLAGS.num_groups,
        int(FLAGS.num_train_examples_per_day / FLAGS.num_groups),
        FLAGS.batch_size)
    test_data = IidDataGenerator(logger, FLAGS.test_data, vocab,
                                 FLAGS.num_groups, 0, FLAGS.batch_size)
  else:
    training_data = NonIidDataGenerator(
        logger, FLAGS.training_data, vocab, FLAGS.num_groups,
        int(FLAGS.num_train_examples_per_day / FLAGS.num_groups), FLAGS.bias,
        FLAGS.batch_size)
    test_data = NonIidDataGenerator(logger, FLAGS.test_data, vocab,
                                    FLAGS.num_groups, 0, FLAGS.bias,
                                    FLAGS.batch_size)
  return vocab, training_data, test_data


def main(unused_args):
  logger = Logger(10)
  log_config(logger)
  vocab, training_data, test_data = init(logger)
  logger.log('Creating model(s)')
  tf.set_random_seed(FLAGS.replica)
  models = []
  if FLAGS.mode == 'sep':
    num_models = FLAGS.num_groups
  else:
    num_models = 1
  for _ in range(num_models):
    model = Model(FLAGS.lr, vocab, FLAGS.bow_limit)
    features, labels, train_op, loss_op, eval_metric_op = model.create_model()
    models.append({
        'model': model,
        'features': features,
        'labels': labels,
        'train_op': train_op,
        'loss_op': loss_op,
        'eval_metric_op': eval_metric_op
    })

  with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for d in range(0, FLAGS.num_days):
      for g in range(0, FLAGS.num_groups):
        if FLAGS.mode == 'sep':
          m = models[g]
        else:
          m = models[0]
        num_train_examples = 0
        # Train.
        for batch in training_data.get(d, g):
          x = m['model'].process_x(batch)
          y = m['model'].process_y(batch)
          num_train_examples = num_train_examples + len(x)
          loss, _ = sess.run([m['loss_op'], m['train_op']], {
              m['features']: x,
              m['labels']: y
          })
          logger.maybe_log('day %d, group %d: trained on %d examples, loss=%f' %
                           (d, g, num_train_examples, loss))
          # Test on all data, mostly for debugging. prefix so we can filter
          # these out.
          if FLAGS.test_after > 0 and (num_train_examples %
                                       FLAGS.test_after) < FLAGS.batch_size:
            for gt in range(0, FLAGS.num_groups):
              test(
                  test_data,
                  m['model'],
                  sess,
                  m['eval_metric_op'],
                  m['features'],
                  m['labels'],
                  logger,
                  d,
                  gt,
                  num_train_examples,
                  prefix='debug ')
        if FLAGS.mode == 'iid':
          for gt in range(0, FLAGS.num_groups):
            test(
                test_data,
                m['model'],
                sess,
                m['eval_metric_op'],
                m['features'],
                m['labels'],
                logger,
                d,
                gt,
                num_train_examples,
                prefix='iid %d on %d: ' % (g, gt))
        elif FLAGS.mode == 'sep':
          for gt in range(0, FLAGS.num_groups):
            test(
                test_data,
                m['model'],
                sess,
                m['eval_metric_op'],
                m['features'],
                m['labels'],
                logger,
                d,
                gt,
                num_train_examples,
                prefix='sep %d on %d: ' % (g, gt))
        elif FLAGS.mode == 'sc':
          for gt in range(0, FLAGS.num_groups):
            test(
                test_data,
                m['model'],
                sess,
                m['eval_metric_op'],
                m['features'],
                m['labels'],
                logger,
                d,
                gt,
                num_train_examples,
                prefix='sc %d on %d: ' % (g, gt))
        else:
          raise ValueError('unsupported mode %s' % FLAGS.mode)
  logger.log('END_MARKER')


if __name__ == '__main__':
  if not sys.version_info >= (3, 0):
    print('This script requires Python3 to run.')
    sys.exit(1)
  app.run(main)
