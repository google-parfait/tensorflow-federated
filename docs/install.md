# Installation

## Install from package

#### 1. Install the python development environment

##### Ubuntu

```bash
sudo apt update
sudo apt install python3-dev python3-pip  # Python 3
sudo pip3 install --upgrade virtualenv  # system-wide install
```

##### macOS

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python  # Python 3
sudo pip3 install --upgrade virtualenv  # system-wide install
```

#### 2. Create a virtual environment

```bash
virtualenv --python python3 "venv"
source "venv/bin/activate"
pip install --upgrade pip
```

#### 3. Install the TensorFlow Federated pip package

```bash
pip install --upgrade tensorflow_federated
```

## Build from source

### Using *pip* on Ubuntu and macOS

#### 1. Install the python developement environment

##### Ubuntu

```bash
sudo apt update
sudo apt install python3-dev python3-pip  # Python 3
sudo pip3 install --upgrade virtualenv  # system-wide install
```

##### macOS

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python  # Python 3
sudo pip3 install --upgrade virtualenv  # system-wide install
```

#### 2. Install Bazel

[Install Bazel](https://docs.bazel.build/versions/master/install.html), the
build tool used to compile Tensorflow Federated.

#### 3. Clone the latest Tensorflow Federated source

```bash
git clone https://github.com/tensorflow/federated.git
cd "federated"
```

#### 4. Create a virtual environment

```bash
virtualenv --python python3 "venv"
source "venv/bin/activate"
pip install --upgrade pip
```

#### 5. Install the Tensorflow Federated package dependencies

```bash
pip install --requirement requirements.txt
```

#### 6. Test Tensorflow Federated

```bash
bazel test \
    --incompatible_remove_native_http_archive=false \
    -- //tensorflow_federated/...
```

#### 7. Exit the virtual environment

```bash
deactivate
```

### Using Docker

#### 1. Install Docker

[Install Docker](https://docs.docker.com/install/) on your local machine.

#### 2. Clone the latest Tensorflow Federated source

```bash
git clone https://github.com/tensorflow/federated.git
```

#### 3. Start a Docker container

```bash
docker run -it \
    --workdir /federated
    --volume ./federated:/federated \
    tensorflow_federated/tensorflow_federated \
    bash
```

#### 4. Test Tensorflow Federated

```bash
bazel test \
    --incompatible_remove_native_http_archive=false \
    -- //tensorflow_federated/...
```
