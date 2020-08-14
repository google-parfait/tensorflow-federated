# Install TensorFlow Federated

There are a few ways to set up your environment to use TensorFlow Federated
(TFF):

*   The easiest way to learn and use TFF requires no installation; run the
    TensorFlow Federated tutorials directly in your browser using
    [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb).
*   To use TensorFlow Federated on a local machine,
    [install the TFF package](#install-tensorflow-federated-using-pip) with
    Python's `pip` package manager.
*   If you have a unique machine configuration,
    [build the TFF package](#build-the-tensorflow-federated-pip-package) from
    source.

## Install TensorFlow Federated using `pip`

### 1. Install the Python development environment.

On Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

On macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

### 2. Create a virtual environment.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Note: To exit the virtual environment, run `deactivate`.

### 3. Install the TensorFlow Federated Python package.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade tensorflow_federated</code>
</pre>

### 4. Test Tensorflow Federated.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

Success: The latest TensorFlow Federated Python package is now installed.

## Build the TensorFlow Federated Python package from source

Building a TensorFlow Federated Python package from source is helpful when you
want to:

*   Make changes to TensorFlow Federated and test those changes in a component
    that uses TensorFlow Federated before those changes are submitted or
    released.
*   Use changes that have been submitted to TensorFlow Federated but have not
    been released.

### 1. Install the Python development environment.

On Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

On macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

### 2. Install Bazel.

[Install Bazel](https://docs.bazel.build/versions/master/install.html), the
build tool used to compile Tensorflow Federated.

### 3. Clone the Tensorflow Federated repository.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/federated.git</code>
<code class="devsite-terminal">cd "federated"</code>
</pre>

### 4. Build the TensorFlow Federated Python package.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/tensorflow_federated"</code>
<code class="devsite-terminal">bazel run //tensorflow_federated/tools/development:build_pip_package -- \
    --nightly \
    --output_dir "/tmp/tensorflow_federated"</code>
</pre>

### 5. Create a new project.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/project"</code>
<code class="devsite-terminal">cd "/tmp/project"</code>
</pre>

### 6. Create a virtual environment.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Note: To exit the virtual environment run `deactivate`.

### 7. Install the TensorFlow Federated Python package.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade "/tmp/tensorflow_federated/"*".whl"</code>
</pre>

### 8. Test Tensorflow Federated.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

Success: A TensorFlow Federated Python package is now built from source and
installed.
