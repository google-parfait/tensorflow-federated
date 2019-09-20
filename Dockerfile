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
FROM ubuntu:18.04

# Install the python development environment
ARG USE_PYTHON_3=True
ARG _PY_SUFFIX=${USE_PYTHON_3:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON}-dev \
    ${PYTHON}-pip \
    git

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -s -f $(which ${PYTHON}) /usr/local/bin/python
RUN ${PYTHON} --version

RUN apt update && apt install -y \
    build-essential \
    curl \
    openjdk-8-jdk \
    pkg-config \
    swig \
    unzip \
    wget \
    g++ \
    zlib1g-dev \
    zip

# Install bazel
ARG BAZEL_VERSION=0.26.1
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh
RUN bazel version

# TODO(b/141279425): Remove pinned tf-estimator-nightly version.
# Install the TensorFlow Federated development environment dependencies
RUN ${PIP} --no-cache-dir install \
    absl-py~=0.7 \
    attrs~=18.2 \
    cachetools~=3.1.1 \
    enum34~=1.1 \
    grpcio~=1.22.0 \
    h5py~=2.6 \
    matplotlib~=3.0 \
    numpy~=1.14 \
    pandas~=0.24.0 \
    portpicker \
    six~=1.10 \
    tensorflow-model-optimization~=0.1.3 \
    tensorflow-privacy~=0.0.1 \
    tf-estimator-nightly==1.14.0.dev2019091601 \
    tf-nightly
RUN pip freeze
