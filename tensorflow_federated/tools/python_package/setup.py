# Copyright 2018, The TensorFlow Federated Authors.
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
"""TensorFlow Federated is an open-source federated learning framework.

TensorFlow Federated (TFF) is an open-source framework for machine learning and
other computations on decentralized data. TFF has been developed to facilitate
open research and experimentation with Federated Learning (FL), an approach to
machine learning where a shared global model is trained across many
participating clients that keep their training data locally. For example, FL has
been used to train prediction models for mobile keyboards without uploading
sensitive typing data to servers.

TFF enables developers to use the included federated learning algorithms with
their models and data, as well as to experiment with novel algorithms. The
building blocks provided by TFF can also be used to implement non-learning
computations, such as aggregated analytics over decentralized data.

TFF's interfaces are organized in two layers:

* Federated Learning (FL) API

  The `tff.learning` layer offers a set of high-level interfaces that allow
  developers to apply the included implementations of federated training and
  evaluation to their existing TensorFlow models.

* Federated Core (FC) API

  At the core of the system is a set of lower-level interfaces for concisely
  expressing novel federated algorithms by combining TensorFlow with distributed
  communication operators within a strongly-typed functional programming
  environment. This layer also serves as the foundation upon which we've built
  `tff.learning`.

TFF enables developers to declaratively express federated computations, so they
could be deployed to diverse runtime environments. Included with TFF is a
single-machine simulation runtime for experiments. Please visit the
tutorials and try it out yourself!
"""
# TODO(b/124800187): Keep in sync with the contents of README.

import datetime
import sys
import setuptools

DOCLINES = __doc__.split('\n')

REQUIRED_PACKAGES = [
    'absl-py~=1.0.0',
    'attrs~=21.2.0',
    'cachetools~=3.1.1',
    'dm-tree~=0.1.1',
    'farmhashpy~=0.4.0',
    'grpcio~=1.34.0',
    'jax~=0.2.27',
    'jaxlib~=0.1.76',
    'numpy~=1.21.4',
    'portpicker~=1.3.1',
    'semantic-version~=2.8.5',
    'tensorflow-model-optimization~=0.7.0',
    'tensorflow-privacy~=0.7.3',
    'tensorflow~=2.6.0',
    'tqdm~=4.28.1',
]

with open('tensorflow_federated/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
  VERSION = globals_dict['__version__']


def get_package_name(requirement: str) -> str:
  allowed_operators = ['~=', '<', '>', '==', '<=', '>=', '!=']
  separator = allowed_operators[0]
  for operator in allowed_operators[1:]:
    requirement = requirement.replace(operator, separator)
  name, _ = requirement.split(separator, maxsplit=1)
  return name


if '--nightly' in sys.argv:
  sys.argv.remove('--nightly')
  PROJECT_NAME = 'tensorflow_federated_nightly'
  date = datetime.date.today().strftime('%Y%m%d')
  VERSION = '{}.dev{}'.format(VERSION, date)
  for index, required_package in enumerate(REQUIRED_PACKAGES):
    package_name = get_package_name(required_package)
    if package_name == 'grpcio':
      REQUIRED_PACKAGES[index] = 'grpcio~=1.37.0'
    elif package_name == 'tensorflow':
      REQUIRED_PACKAGES[index] = 'tf-nightly'
else:
  PROJECT_NAME = 'tensorflow_federated'

setuptools.setup(
    name=PROJECT_NAME,
    version=VERSION,
    packages=setuptools.find_packages(exclude=('tools')),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type='text/plain',
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    url='http://tensorflow.org/federated',
    download_url='https://github.com/tensorflow/federated/tags',
    install_requires=REQUIRED_PACKAGES,
    # PyPI package information.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='tensorflow federated machine learning',
)
