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
"""A generic worker binary for deployment, e.g., on GCP."""

from absl import app
from absl import flags
import grpc
import tensorflow as tf

from tensorflow_federated.python.core import framework
from tensorflow_federated.python.simulation import server_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('port', '8000', 'port to listen on')
flags.DEFINE_integer('threads', '10', 'number of worker threads in thread pool')
flags.DEFINE_string('private_key', '', 'the private key for SSL/TLS setup')
flags.DEFINE_string('certificate_chain', '', 'the cert for SSL/TLS setup')


def main(argv):
  del argv
  tf.compat.v1.enable_v2_behavior()
  executor = framework.create_local_executor()(None)
  if FLAGS.private_key:
    if FLAGS.certificate_chain:
      with open(FLAGS.private_key, 'rb') as f:
        private_key = f.read()
      with open(FLAGS.certificate_chain, 'rb') as f:
        certificate_chain = f.read()
      credentials = grpc.ssl_server_credentials(((
          private_key,
          certificate_chain,
      ),))
    else:
      raise ValueError(
          'Private key has been specified, but the certificate chain missing.')
  else:
    credentials = None
  server_utils.run_server(executor, FLAGS.threads, FLAGS.port, credentials)


if __name__ == '__main__':
  app.run(main)
