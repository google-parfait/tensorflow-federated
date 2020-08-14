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
"""Script for testing a remote executor on GCP."""

from absl import app
from absl import flags
import grpc
import tensorflow_federated as tff

FLAGS = flags.FLAGS

flags.DEFINE_string('host', None, 'The host to connect to.')
flags.mark_flag_as_required('host')
flags.DEFINE_string('port', '8000', 'The port to connect to.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  channel = grpc.insecure_channel('{}:{}'.format(FLAGS.host, FLAGS.port))
  ex = tff.framework.RemoteExecutor(channel)
  ex = tff.framework.CachingExecutor(ex)
  ex = tff.framework.ReferenceResolvingExecutor(ex)
  factory = tff.framework.create_executor_factory(lambda _: ex)
  context = tff.framework.ExecutionContext(factory)
  tff.framework.set_default_context(context)

  print(tff.federated_computation(lambda: 'Hello World')())


if __name__ == '__main__':
  app.run(main)
