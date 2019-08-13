# Lint as: python3
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

import sys

import setuptools

DOCLINES = __doc__.split('\n')

_VERSION = '0.7.0'

project_name = 'tensorflow_federated'

if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)

REQUIRED_PACKAGES = [
    'attrs~=18.2',
    'cachetools~=3.1.1',
    'h5py~=2.6',
    'numpy~=1.14',
    'six~=1.10',
    'tf-nightly==1.15.0.dev20190805',
    'grpcio~=1.22.0',
    'portpicker',
    'tensorflow-model-optimization~=0.1.2',
]

setuptools.setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
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
    classifiers=(
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license='Apache 2.0',
    keywords='tensorflow federated machine learning',
)
