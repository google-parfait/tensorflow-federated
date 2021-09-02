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
"""An aggregator service for testing."""

import asyncio
import functools
import signal

from absl import app
from absl import flags
from absl import logging
import grpc
import tensorflow_federated as tff

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'worker_port',
    10000,
    lower_bound=1024,
    upper_bound=65535,
    help='The port to listen on.')
flags.DEFINE_integer(
    'aggregator_port',
    10001,
    lower_bound=1024,
    upper_bound=65535,
    help='The port to listen on.')

GRPC_MAX_MESSAGE_LENGTH_BYTES = 1024 * 1024 * 1024
GRPC_CHANNEL_OPTIONS = [
    ('grpc.max_message_length', GRPC_MAX_MESSAGE_LENGTH_BYTES),
    ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_LENGTH_BYTES),
    ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH_BYTES)
]


def handler(shutdown_event):
  logging.info('**** Received SIGINT, initiating shutdown...')
  shutdown_event.set()
  logging.info('**** Shutdown even set.')


async def wait(shutdown_event):
  logging.info('**** Running server...')
  await shutdown_event.wait()
  logging.info('**** Exiting server...')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  event_loop = asyncio.get_event_loop()

  # Catch SIGINT, which the test framework will raise.
  shutdown_event = asyncio.Event()
  event_loop.add_signal_handler(
      signal.SIGINT, functools.partial(handler, shutdown_event=shutdown_event))

  server_endpoint = f'[::]:{FLAGS.worker_port}'
  insecure_channel = grpc.insecure_channel(
      server_endpoint, options=GRPC_CHANNEL_OPTIONS)
  ex_context = tff.backends.native.create_remote_python_execution_context(
      channels=[insecure_channel])

  with tff.simulation.server_context(
      ex_context.executor_factory,
      num_threads=1,
      port=FLAGS.aggregator_port,
      options=GRPC_CHANNEL_OPTIONS):
    event_loop.run_until_complete(wait(shutdown_event))
  logging.info('Exiting process')


if __name__ == '__main__':
  app.run(main)
