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
FROM ubuntu:16.04

# Install the python developement environment
ARG USE_PYTHON_3
ARG _PY_SUFFIX=${USE_PYTHON_3:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}
RUN test ${USE_PYTHON_3} && ln -s /usr/bin/python3 /usr/local/bin/python || true
RUN apt-get update && apt-get install -y \
    ${PYTHON}-dev \
    ${PYTHON}-pip
RUN ${PIP} install --upgrade \
    pip \
    setuptools

# Install Bazel
RUN apt update && apt install -y \
    curl \
    openjdk-8-jdk
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | tee "/etc/apt/sources.list.d/bazel.list"
RUN curl https://bazel.build/bazel-release.pub.gpg \
    | apt-key add -
RUN apt update && apt install -y bazel

# Install the Tensorflow Federated package dependencies
RUN ${PIP} install --no-cache-dir \
    enum34 \
    keras_applications \
    keras_preprocessing \
    mock \
    numpy \
    six \
    wheel
