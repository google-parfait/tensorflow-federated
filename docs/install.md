# Install TensorFlow Federated

The easiest way to learn and use TensorFlow Federated requires no installation!
You can get started by running the TensorFlow Federated [tutorials](tutorials/)
directly in a brownser using
[Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb).

If you want to use TensorFlow Federated locally on your machine you can
[install TensorFlow Federated](#install-tensorFlow-federated-using-pip) using
Python's pip package manager.

If you have a unique machine configuration you can
[build TensorFlow Federated](#build-the-tensorflow-federated-pip-package) from
source.

## Install TensorFlow Federated using pip

Install TensorFlow Federated using Python's pip package manager on Ubuntu or
macOS.

1.  Install the python development environment.

    Ubuntu

    ```bash
    sudo apt update
    sudo apt install python3-dev python3-pip  # Python 3
    sudo pip3 install --upgrade virtualenv  # system-wide install
    ```

    macOS

    ```bash
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
    brew update
    brew install python  # Python 3
    sudo pip3 install --upgrade virtualenv  # system-wide install
    ```

1.  Create a virtual environment.

    ```bash
    virtualenv --python python3 "venv"
    source "venv/bin/activate"
    pip install --upgrade pip
    ```

    NOTE: To exit the virtual environment run `deactivate`.

1.  Install the pip package.

    ```bash
    pip install --upgrade tensorflow_federated
    ```

1.  Test Tensorflow Federated.

    ```bash
    python -c "import tensorflow_federated as tff; tff.federated_computation(lambda: 'Hello, World!')()"
    ```

## Build the TensorFlow Federated pip package

Build the TensorFlow Federated pip package and install it on Ubuntu or macOS.

1.  Create a development environment [using virtualenv](#using-virtualenv) or
    [using Docker](#using_docker).

1.  [Build the pip package](#build-the-pip-package).

### Using `virtualenv`

Create a Tensorflow Federated development environment using `virtualenv` on
Ubuntu or macOS.

1.  Install the Python development environment.

    Ubuntu

    ```bash
    sudo apt update
    sudo apt install python3-dev python3-pip  # Python 3
    sudo pip3 install --upgrade virtualenv  # system-wide install
    ```

    macOS

    ```bash
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
    brew update
    brew install python  # Python 3
    sudo pip3 install --upgrade virtualenv  # system-wide install
    ```

1.  Install Bazel.

    [Install Bazel](https://docs.bazel.build/versions/master/install.html), the
    build tool used to compile Tensorflow Federated.

    NOTE: Bazel version `0.19.2` or greater is required by TensorFlow Federated.

1.  Clone the Tensorflow Federated repository.

    ```bash
    git clone https://github.com/tensorflow/federated.git
    cd "federated"
    ```

1.  Create a virtual environment.

    ```bash
    virtualenv --python python3 "venv"
    source "venv/bin/activate"
    pip install --upgrade pip
    ```

    NOTE: To exit the virtual environment run `deactivate`.

1.  Install Tensorflow Federated dependencies.

    ```bash
    pip install --requirement requirements.txt
    ```

1.  Test Tensorflow Federated.

    ```bash
    bazel test //tensorflow_federated/...
    ```

### Using Docker

Create a Tensorflow Federated development environment using Docker on Ubuntu or
macOS.

1.  Install Docker.

    [Install Docker](https://docs.docker.com/install/) on your local machine.

1.  Clone the latest Tensorflow Federated source.

    ```bash
    git clone https://github.com/tensorflow/federated.git
    ```

1.  Start a Docker container.

    ```bash
    docker run -it \
        --workdir /federated
        --volume ./federated:/federated \
        tensorflow_federated/tensorflow_federated \
        bash
    ```

1.  Test Tensorflow Federated.

    ```bash
    bazel test //tensorflow_federated/...
    ```

### Build the pip package

Build the TensorFlow Federated pip package and install it on Ubuntu or macOS.

1.  Build the pip package.

    ```bash
    mkdir "/tmp/tensorflow_federated"
    bazel build //tensorflow_federated/tools:build_pip_package
    bazel-bin/tensorflow_federated/tools/build_pip_package "/tmp/tensorflow_federated"
    ```

1.  Create a new project.

    ```bash
    mkdir "/tmp/project"
    cd "/tmp/project"

    virtualenv --python python3 "venv"
    source "venv/bin/activate"
    pip install --upgrade pip
    ```

    NOTE: To exit the virtual environment run `deactivate`.

1.  Install the pip package.

    ```bash
    pip install --upgrade "/tmp/tensorflow_federated/tensorflow_federated-"*".whl"
    ```

1.  Test Tensorflow Federated.

    ```bash
    python -c "import tensorflow_federated as tff; tff.federated_computation(lambda: 'Hello, World!')()"
    ```
