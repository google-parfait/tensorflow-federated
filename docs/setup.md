# Installation

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
virtualenv --system-site-packages --python python3 "venv"
source "venv/bin/activate"
pip install --upgrade pip
```

#### 5. Install the Tensorflow Federated package dependencies

```bash
pip install --requirement requirements.txt
```

#### 6. Test Tensorflow Federated

```bash
bazel test ...
```

#### 7. Exit the virtual environment

```bash
deactivate
```

### Using Docker on Ubuntu

#### 1. Install Docker

[Install Docker](https://docs.docker.com/install/) on your local machine.

#### 2. Install Bazel

[Install Bazel](https://docs.bazel.build/versions/master/install.html), the
build tool used to compile Tensorflow Federated.

#### 3. Clone the latest Tensorflow Federated source

```bash
git clone https://github.com/tensorflow/federated.git
```

#### 4. Start a Docker container

```bash
docker run -it \
    --workdir /federated
    --volume ./federated:/federated \
    tensorflow_federated/tensorflow_federated \
    bash
```

#### 5. Test Tensorflow Federated

```bash
bazel test ...
```
