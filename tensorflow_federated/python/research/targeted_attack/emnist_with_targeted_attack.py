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
"""Script for federated training of EMNIST classifiers with targeted attack and corrsponding defenses."""

import collections
import os

from absl import app
from absl import flags
import numpy as np
from scipy import io
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.targeted_attack import aggregate_fn
from tensorflow_federated.python.research.targeted_attack import attacked_fedavg

FLAGS = flags.FLAGS

# training parameters
flags.DEFINE_string('root_output_dir', '/tmp/emnist_grids/',
                    'Root directory for writing experiment output.')
flags.DEFINE_integer('random_seed', 0, 'Random seed.')
flags.DEFINE_integer('evaluate_per_rounds', 1, 'Clients number per round.')
flags.DEFINE_boolean('only_digits', True, 'True: 10 classes, False 62 classes.')

# server parameters
flags.DEFINE_integer('num_clients_per_round', 5, 'Clients number per round.')
flags.DEFINE_integer('num_rounds', 300, 'Number of rounds.')
flags.DEFINE_float('server_learning_rate', 1., 'Server learning rate.')
flags.DEFINE_float('server_momentum', 0., 'Server learning rate.')

# client parameters
flags.DEFINE_integer('num_epochs', 5, 'Number of epochs in the client.')
flags.DEFINE_integer('batch_size', 20, 'Training batch size.')
flags.DEFINE_float('client_learning_rate', 0.1, 'Client learning rate.')
flags.DEFINE_float('client_momentum', 0., 'Client learning rate.')

# attack parameters
flags.DEFINE_integer('attack_freq', 1, 'Attacking frequency of the attacker.')
flags.DEFINE_integer('task_num', 30,
                     'The number of attack tasks we want to insert.')
flags.DEFINE_integer(
    'client_round_num', 5,
    'Number of local rounds used to compute the malicious update.')

# defense parameters
flags.DEFINE_float('drop_prob', 0.5, 'Dropping probability of each layer')
flags.DEFINE_float('norm_bound', 0.33,
                   'The maximum norm for malicious update before boosting.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'The clipped l2 norm')
flags.DEFINE_float('mul_factor', 0.,
                   'The multiplication factor to ensure privacy')

keys = [
    'random_seed', 'num_clients_per_round', 'num_rounds',
    'server_learning_rate', 'num_epochs', 'batch_size', 'client_learning_rate',
    'client_round_num', 'attack_freq', 'task_num', 'drop_prob', 'norm_bound',
    'l2_norm_clip', 'mul_factor'
]

use_nchw_format = False
data_format = 'channels_first' if use_nchw_format else 'channels_last'
data_shape = [1, 28, 28] if use_nchw_format else [28, 28, 1]


def preprocess(dataset):
  """Preprocess dataset."""

  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], data_shape)),
        ('y', tf.reshape(element['label'], [-1])),
    ])

  return dataset.repeat(FLAGS.num_epochs).map(element_fn).batch(
      FLAGS.batch_size)


def load_malicious_dataset(num_tasks):
  """Load malicious dataset consisting of malicious target samples."""
  url_malicious_dataset = 'https://storage.googleapis.com/tff-experiments-public/targeted_attack/emnist_malicious/emnist_target.mat'
  filename = 'emnist_target.mat'
  path = tf.keras.utils.get_file(filename, url_malicious_dataset)
  emnist_target_data = io.loadmat(path)
  emnist_target_x = emnist_target_data['target_train_x'][0]
  emnist_target_y = emnist_target_data['target_train_y'][0]
  target_x = np.concatenate(emnist_target_x[-num_tasks:], axis=0)
  target_y = np.concatenate(emnist_target_y[-num_tasks:], axis=0)
  dict_malicious = collections.OrderedDict([('x', target_x), ('y', target_y)])
  dataset_malicious = tf.data.Dataset.from_tensors(dict_malicious)
  return dataset_malicious, target_x, target_y


def load_test_data():
  """Load test data for faster evaluation."""
  url_test_data = 'https://storage.googleapis.com/tff-experiments-public/targeted_attack/emnist_test_data/emnist_test_data.mat'
  filename = 'emnist_test_data.mat'
  path = tf.keras.utils.get_file(filename, url_test_data)
  emnist_test_data = io.loadmat(path)
  test_image = emnist_test_data['test_x']
  test_label = emnist_test_data['test_y']
  return test_image, test_label


def make_federated_data_with_malicious(client_data,
                                       dataset_malicious,
                                       client_ids,
                                       with_attack=1):
  """Make federated dataset with potential attackers."""
  benign_dataset = [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]
  malicious_dataset = [dataset_malicious for x in client_ids]
  if with_attack:
    client_type_list = \
        [tf.cast(0, tf.bool)] * (len(client_ids)-1) + [tf.cast(1, tf.bool)]
  else:
    client_type_list = [tf.cast(0, tf.bool)] * len(client_ids)
  return benign_dataset, malicious_dataset, client_type_list


def sample_clients_with_malicious(client_data,
                                  client_ids,
                                  dataset_malicious,
                                  num_clients=3,
                                  with_attack=1):
  """Sample client and make federated dataset."""
  sampled_clients = np.random.choice(client_ids, num_clients)
  federated_train_data, federated_malicious_data, client_type_list = \
      make_federated_data_with_malicious(client_data, dataset_malicious,
                                         sampled_clients, with_attack)
  return federated_train_data, federated_malicious_data, client_type_list


def create_keras_model():
  """Build compiled keras model."""
  num_classes = 10 if FLAGS.only_digits else 62
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          input_shape=data_shape,
          data_format=data_format),
      tf.keras.layers.Conv2D(
          64, kernel_size=(3, 3), activation='relu', data_format=data_format),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(num_classes, activation='softmax')
  ])
  return model


def evaluate(state, x, y, target_x, target_y, batch_size=100):
  """Evaluate the model on both main task and target task."""
  keras_model = create_keras_model()
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  tff.learning.assign_weights_to_keras_model(keras_model, state.model)
  test_metrics = keras_model.evaluate(x, y, batch_size=batch_size)
  test_metrics_target = keras_model.evaluate(
      target_x, target_y, batch_size=batch_size)
  return test_metrics, test_metrics_target


def write_print(file_handle, line):
  print(line)
  file_handle.write(line + '\n')


def log_tfboard(name, value, step):
  tf.summary.scalar(name, value, step=step)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = tf.compat.v1.ConfigProto()
  config.graph_options.rewrite_options.layout_optimizer = 2
  tf.compat.v1.enable_eager_execution(config)

  np.random.seed(FLAGS.random_seed)

  flag_dict = FLAGS.flag_values_dict()
  configs = '-'.join(
      ['{}={}'.format(k, flag_dict[k]) for k in keys if k != 'root_output_dir'])
  file_name = 'log' + configs
  file_handle = open(os.path.join(FLAGS.root_output_dir, file_name), 'w')

  global_step = tf.Variable(1, name='global_step', dtype=tf.int64)
  file_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.root_output_dir))
  file_writer.set_as_default()
  write_print(file_handle, '=======configurations========')
  write_print(file_handle, configs)
  write_print(file_handle, '=======configurations========')
  # prepare dataset.
  write_print(file_handle, 'Loading Dataset!')
  emnist_train, _ = tff.simulation.datasets.emnist.load_data(
      only_digits=FLAGS.only_digits)

  # prepare test set
  write_print(file_handle, 'Loading Test Set!')
  test_image, test_label = load_test_data()

  # load malicious dataset
  write_print(file_handle, 'Loading malicious dataset!')
  dataset_malicious, target_x, target_y = load_malicious_dataset(FLAGS.task_num)

  # prepare model_fn.
  example_dataset = emnist_train.create_tf_dataset_for_client(
      emnist_train.client_ids[0])
  sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                       iter(preprocess(example_dataset)).next())

  def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        dummy_batch=sample_batch,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  # define server optimizer
  nesterov = True if FLAGS.server_momentum != 0 else False

  def server_optimizer_fn():
    return tf.keras.optimizers.SGD(
        learning_rate=FLAGS.server_learning_rate,
        momentum=FLAGS.server_momentum,
        nesterov=nesterov)

  # build interative process
  write_print(file_handle, 'Building Iterative Process!')
  client_update_function = attacked_fedavg.ClientProjectBoost(
      boost_factor=float(FLAGS.num_clients_per_round),
      norm_bound=FLAGS.norm_bound,
      round_num=FLAGS.client_round_num)
  aggregation_function = aggregate_fn.build_dp_aggregate(
      l2_norm=FLAGS.l2_norm_clip,
      mul_factor=FLAGS.mul_factor,
      num_clients=FLAGS.num_clients_per_round)

  iterative_process = attacked_fedavg.build_federated_averaging_process_attacked(
      model_fn=model_fn,
      stateful_delta_aggregate_fn=aggregation_function,
      client_update_tf=client_update_function,
      server_optimizer_fn=server_optimizer_fn)
  state = iterative_process.initialize()

  # training loop
  for cur_round in range(FLAGS.num_rounds):
    if cur_round % FLAGS.attack_freq == FLAGS.attack_freq // 2:
      with_attack = 1
      write_print(file_handle, 'Attacker appears!')
    else:
      with_attack = 0

    # sample clients and make federated dataset
    federated_train_data, federated_malicious_data, client_type_list = \
        sample_clients_with_malicious(
            emnist_train, client_ids=emnist_train.client_ids,
            dataset_malicious=dataset_malicious,
            num_clients=FLAGS.num_clients_per_round, with_attack=with_attack)

    # one round of attacked federated averaging
    write_print(file_handle, 'Round starts!')
    state, train_metrics = iterative_process.next(state, federated_train_data,
                                                  federated_malicious_data,
                                                  client_type_list)

    write_print(
        file_handle,
        'Training round {:2d}, train_metrics={}'.format(cur_round,
                                                        train_metrics))

    log_tfboard('train_acc', train_metrics[0], global_step)
    log_tfboard('train_loss', train_metrics[1], global_step)

    # evaluate current model on test data and malicious data
    if cur_round % FLAGS.evaluate_per_rounds == 0:
      test_metrics, test_metrics_target = evaluate(state, test_image,
                                                   test_label, target_x,
                                                   target_y)
      write_print(
          file_handle,
          'Evaluation round {:2d}, <sparse_categorical_accuracy={},loss={}>'
          .format(cur_round, test_metrics[1], test_metrics[0]))
      write_print(
          file_handle,
          'Evaluation round {:2d}, <sparse_categorical_accuracy={},loss={}>'
          .format(cur_round, test_metrics_target[1], test_metrics_target[0]))
      log_tfboard('test_acc', test_metrics[1], global_step)
      log_tfboard('test_loss', test_metrics[0], global_step)
      log_tfboard('test_acc_target', test_metrics_target[1], global_step)
      log_tfboard('test_loss_target', test_metrics_target[0], global_step)

    global_step.assign_add(1)


if __name__ == '__main__':
  app.run(main)
