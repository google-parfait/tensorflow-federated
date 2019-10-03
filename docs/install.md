# Install TensorFlow Federated

There are a few ways to set up your environment to use TensorFlow Federated
(TFF):

*   The easiest way to learn and use TFF requires no installation—run the
    TensorFlow Federated tutorials directly in your browser using
    [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb).
*   To use TensorFlow Federated on a local machine, install the
    [TFF package](#install-tensorflow-federated-using-pip) with Python's `pip`
    package manager.
*   If you have a unique machine configuration,
    [build TensorFlow Federated](#build-the-tensorflow-federated-pip-package)
    from source.

## Install TensorFlow Federated using pip

#### 1. Install the Python development environment.

On Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

On macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. Create a virtual environment.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Note: To exit the virtual environment, run `deactivate`.

#### 3. Install the TensorFlow Federated `pip` package.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade tensorflow_federated</code>
</pre>

#### 4. (Optional) Test Tensorflow Federated.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

Success: TensorFlow Federated is now installed.

## Build the TensorFlow Federated pip package

### 1. Install the Python development environment.

On Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

On macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

### 2. Install Bazel.

[Install Bazel](https://docs.bazel.build/versions/master/install.html), the
build tool used to compile Tensorflow Federated.

### 3. Clone the Tensorflow Federated repository.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/federated.git</code>
<code class="devsite-terminal">cd "federated"</code>
</pre>

### 4. Create a virtual environment.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Note: To exit the virtual environment, run `deactivate`.

### 5. Install Tensorflow Federated dependencies.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --requirement "requirements.txt"</code>
</pre>

### 6. (Optional) Test Tensorflow Federated.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">bazel test //tensorflow_federated/...</code>
</pre>

#### 7. Build the pip package.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/tensorflow_federated"</code>
<code class="devsite-terminal">bazel run //tensorflow_federated/tools/development:build_pip_package -- \
    "/tmp/tensorflow_federated"</code>
</pre>

#### 8. Create a new project.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/project"</code>
<code class="devsite-terminal">cd "/tmp/project"</code>

<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Note: To exit the virtual environment run `deactivate`.

#### 9. Install the pip package.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade "/tmp/tensorflow_federated/tensorflow_federated-"*".whl"</code>
</pre>

#### 10. Test Tensorflow Federated.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

Success: The TensorFlow Federated package is built.

## Using Docker

Create a Tensorflow Federated development environment using Docker on Ubuntu or
macOS.

### 1. Install Docker.

[Install Docker](https://docs.docker.com/install/) on your local machine.

### 2. Clone the latest Tensorflow Federated source.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/federated.git</code>
<code class="devsite-terminal">cd "federated"</code>
</pre>

### 3. Build a Docker image.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">docker build . \
    --tag tensorflow_federated:latest</code>
</pre>

### 4. Start a Docker container.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">docker run -it \
    --workdir /federated \
    --volume $(pwd):/federated \
    tensorflow_federated:latest \
    bash</code>
</pre>

### 5. (Optional) Test Tensorflow Federated.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">bazel test //tensorflow_federated/...</code>
</pre>

Success: The TensorFlow Federated development environment is ready, now
[build the pip package](#build-the-pip-package).
