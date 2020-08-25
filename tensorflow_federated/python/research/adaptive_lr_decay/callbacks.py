# Copyright 2020, The TensorFlow Federated Authors.
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
"""Library of TFF learning rate callbacks."""

import attr
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


@attr.s(eq=False)
class ReduceLROnPlateau(object):
  """A callback for decaying a learning rate when a metric stops improving.

  Attributes:
    learning_rate: The current learning rate.
    monitor: The name of the metric governing the callback.
    decay_factor: Factor by which the learning rate will be reduced when a
      plateau occurs.
    minimize: A boolean, when `True` the metric will be minimized, otherwise
      maximized.
    best: The previous best metric result. The initial value of this should be
      set so that any reasonable metric improves upon the best. For example, if
      `minimize = True`, then `best = np.Inf` serves as a reasonable initial
      value.
    min_delta: The amount by which the monitored metric must improve upon the
      best previous metric in order for there to be definite improvement. For
      example, if we set `minimize=True`, then a new value of the metric is said
      to be better than the previous best value only if `metric <= best -
      min_delta`.
    min_lr: The minimum learning rate, beyond which no decay is allowed.
    window_size: The number of rounds to average the metric across. In federated
      learning, we typically do not have access to metrics computed over the
      entire training set at any given round. Instead, we compute the average of
      the monitored metric over the last `window_size` rounds, and use this as
      an estimate of the true value of the monitored metric over the entire
      training set.
    metrics_window: A list of the value of the metric over the last
      `window_size` rounds. This should be initialized according to the
      `minimize` argument, similar to how `best` is initialized. For example, if
      `minimize = True`, then initializing `metrics_window` to be a list of
      `np.Inf` values of length `window_size` is a reasonable initialization.
    patience: The number of rounds that must occur with no improvement before
      the learning rate can be reduced. This parameter is used to ensure that
      learning rate decay does not occur too frequently. This parameter is
      independent but related to the `window_size` argument. In general, one
      should set `patience` to be at least on the order of `window_size` in
      order to ensure that the plateau is not an artifact of the client
      selection over some small number of rounds. A good starting heuristic is
      `patience = window_size`.
    wait: The number of rounds that have passed with no improvement.
    cooldown: The number of rounds that must occur before reducing the learning
      rate. This should
      generally be at least as large as `window_size`, so that enough rounds
      pass to allow metrics_window to be an accurate estimate of the current
      metric.
    cooldown_counter: The number of rounds remaining in the cooldown period.
      Resets after the learning rate has been decayed. While in the cooldown
      period, the `wait` parameter is set to 0. Thus, the total number of rounds
      before the learning rate decays is at least `cooldown + patience`.
  """

  learning_rate = attr.ib()
  monitor = attr.ib(default='loss')
  decay_factor = attr.ib(default=0.1)
  minimize = attr.ib(default=True)
  best = attr.ib(default=None)
  min_delta = attr.ib(default=1e-4)
  min_lr = attr.ib(default=0.0)
  window_size = attr.ib(default=100)
  metrics_window = attr.ib(default=None)
  patience = attr.ib(default=100)
  wait = attr.ib(default=0)
  cooldown = attr.ib(default=None)
  cooldown_counter = attr.ib(default=None)

  def update(self, round_metric):
    """Updates the `ReduceLROnPlateau` callback based on the round metric."""
    metrics_window = self.metrics_window[1:]
    metrics_window.append(round_metric)
    average_metric = tf.reduce_mean(metrics_window)

    learning_rate = self.learning_rate
    best = self.best
    wait = self.wait
    cooldown_counter = self.cooldown_counter

    if cooldown_counter > 0:
      cooldown_counter -= 1
      wait = 0

    if self.improves_best(average_metric):
      best = average_metric
      wait = 0
    elif cooldown_counter <= 0:
      wait += 1
      if wait >= self.patience:
        if learning_rate > self.min_lr:
          learning_rate = tf.maximum(learning_rate * self.decay_factor,
                                     self.min_lr)
          cooldown_counter = self.cooldown
          wait = 0

    # Return an updated callback
    return tff.utils.update_state(
        self,
        learning_rate=learning_rate,
        metrics_window=metrics_window,
        best=best,
        wait=wait,
        cooldown_counter=cooldown_counter)

  def improves_best(self, metric):
    """Determines if a round metric improves a given ReduceLROnPlateau`."""
    if self.minimize and metric < self.best - self.min_delta:
      return True
    elif not self.minimize and metric > self.best + self.min_delta:
      return True
    else:
      return False


def create_reduce_lr_on_plateau(**kwargs):
  """Initializes a callback in a way that automatically infers attributes."""
  callback = ReduceLROnPlateau(**kwargs)
  if callback.learning_rate < callback.min_lr:
    callback.learning_rate = callback.min_lr

  if callback.decay_factor > 1.0 or callback.decay_factor < 0:
    raise ValueError('Decay factor must be in the range [0, 1].')

  if callback.minimize not in [True, False]:
    raise ValueError('The attribute minimize must be True or False.')

  if callback.best is None:
    callback.best = np.Inf if callback.minimize else 0.0

  if callback.min_delta < 0.0:
    raise ValueError('min_delta must be nonnegative.')

  if callback.metrics_window is None:
    if callback.minimize:
      callback.metrics_window = [np.Inf for _ in range(callback.window_size)]
    else:
      callback.metrics_window = [0.0 for _ in range(callback.window_size)]

  if len(callback.metrics_window) != callback.window_size:
    raise ValueError(
        'The metrics window must be of length {}, received a window of length'
        ' {}'.format(callback.window_size, len(callback.metrics_window)))

  if callback.cooldown is None:
    callback.cooldown = callback.window_size

  if callback.cooldown_counter is None:
    callback.cooldown_counter = callback.cooldown

  return callback
