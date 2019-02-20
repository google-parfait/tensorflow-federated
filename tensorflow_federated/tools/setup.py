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
# pylint: disable=line-too-long
"""TensorFlow Federated is an open-source federated learning framework.

TensorFlow Federated (TFF) is an open-source framework for collaborative
computations on distributed data that does not require collecting data at a
centralized location.

The framework has initially been developed to facilitate open research and
experimentation with
[Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html),
a technology that enables devices owned by end users to collaboratively learn a
shared prediction model while keeping potentially sensitive training data on the
devices, thus decoupling the ability to do machine learning from the need to
collect and store the data in the cloud.

With the interfaces provided by TFF, developers can test existing federated
learning algorithms on their models and data, or design new experimental
algorithms and run them on existing models and data, all within the same open
source environment. The framework has been designed with compositionality in
mind, and can be used to combine independently-developed techniques and
components that offer complementary capabilities into larger systems.
"""
# pylint: enable=line-too-long
# TODO(b/124800187): Keep in sync with the contents of README.

import sys

import setuptools

DOCLINES = __doc__.split('\n')

_VERSION = '0.1.0'

project_name = 'tensorflow_federated'

# Set when building the pip package
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)

REQUIRED_PACKAGES = [
    'h5py',
    'numpy',
    'six',
    'tensorflow>=1.13.0rc2',
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
