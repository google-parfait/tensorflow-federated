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
"""Tool to generate external API documentation for tensorflow_federated."""

import inspect
import os

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib

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


def generate_api_docs(output_dir):
  """Generates markdown API docs for TFF.

  Args:
    output_dir: Base directory path to write generated files to.
  """

  def _ignore_symbols(mod):
    """Returns list of symbols to ignore for documentation."""
    all_symbols = [x for x in dir(mod) if not x.startswith('_')]
    allowed_symbols = mod._allowed_symbols  # pylint: disable=protected-access
    return set(all_symbols) - set(allowed_symbols)

  def _get_ignored_symbols(module):
    """Returns a Python `set` of symbols to ignore for a given `module`."""
    symbols = dir(module)
    private_symbols = [x for x in symbols if x.startswith('_')]
    module_symbols = [
        x for x in symbols if inspect.ismodule(getattr(module, x))
    ]
    return set(private_symbols + module_symbols)

  doc_generator = generate_lib.DocGenerator(
      root_title='TensorFlow Federated',
      py_modules=[('tff', tff)],
      base_dir=os.path.dirname(tff.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      private_map={
          'tff':
              _ignore_symbols(tff),
          'tff.backends':
              _ignore_symbols(tff.backends),
          'tff.backends.mapreduce':
              _ignore_symbols(tff.backends.mapreduce),
          'tff.framework':
              _ignore_symbols(tff.framework),
          'tff.learning':
              _ignore_symbols(tff.learning),
          'tff.learning.framework':
              _ignore_symbols(tff.learning.framework),
          'tff.simulation':
              _ignore_symbols(tff.simulation),
          'tff.simulation.datasets':
              _ignore_symbols(tff.simulation.datasets),
          'tff.simulation.datasets.cifar100':
              _get_ignored_symbols(tff.simulation.datasets.cifar100),
          'tff.simulation.datasets.emnist':
              _get_ignored_symbols(tff.simulation.datasets.emnist),
          'tff.simulation.datasets.shakespeare':
              _get_ignored_symbols(tff.simulation.datasets.shakespeare),
          'tff.simulation.datasets.stackoverflow':
              _get_ignored_symbols(tff.simulation.datasets.stackoverflow),
          'tff.simulation.models':
              _ignore_symbols(tff.simulation.models),
          'tff.simulation.models.mnist':
              _get_ignored_symbols(tff.simulation.models.mnist),
          'tff.templates':
              _ignore_symbols(tff.templates),
          'tff.test':
              _ignore_symbols(tff.test),
          'tff.utils':
              _ignore_symbols(tff.utils),
      })
  doc_generator.build(output_dir)


def main(unused_argv):
  generate_api_docs(FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
