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
"""Tool to generate external API documentation for tensorflow_federated."""

import os

from absl import app
from absl import flags
import tensorflow_docs
import tensorflow_federated as tff

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', '/tmp/federated_api',
                    'Where to output the docs')

flags.DEFINE_string('code_url_prefix', None,
                    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', 'federated/api_docs/python',
                    'Path prefix in the _toc.yaml')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  doc_generator = tensorflow_docs.api_generator.generate_lib.DocGenerator(
      root_title='TensorFlow Federated',
      py_modules=[('tff', tff)],
      base_dir=os.path.dirname(tff.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      callbacks=[
          tensorflow_docs.api_generator.public_api
          .explicit_package_contents_filter,
      ])
  doc_generator.build(FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
