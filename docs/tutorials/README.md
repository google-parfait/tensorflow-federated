# TensorFlow Federated Tutorials

## Running the TFF Tutorials in Jupyter Notebooks

By design [asyncio](https://docs.python.org/3/library/asyncio.html) does not
allow its event loop to be nested. This presents a practical problem: When in an
environment where the event loop is already running it’s impossible to run tasks
and wait for the result. Trying to do so will give the error `“RuntimeError:
This event loop is already running”`.

This issue is present when running the TFF tutorials in Google Colab or Jupyter
notebooks.

The fix is to:

1.  Install [nest-asyncio](https://pypi.org/project/nest-asyncio/) by either:

    *   `pip` install the package from within the notebook.

        ```python
        !pip install --quiet --upgrade --user nest-asyncio
        ```

    *   `pip` install the package in the same virtual environment that the
        IPython kernel is installed.

        ```python
        # Create a virtual environment
        python3 -m venv "venv"
        source "venv/bin/activate"
        pip install --upgrade "pip==20.2.4"

        # Install the required Python package
        pip install --upgrade \
            tensorflow-federated \
            nest-asyncio

        # Install Jupyter
        pip install --upgrade jupyter

        # Build IPython kernel
        python -m ipykernel install \
            --user \
            --name tff_kernel \
            --display-name "TFF Kernel"
        ```

        <!-- TODO(b/242107901): Downgraded pip due to bug installing compatible
        versions of required packages https://github.com/pypa/pip/issues/9613.
        As of 8/10/22 this is the latest version that works. pip 20.3.x was
        verified to fail.
        -->

1.  Use `nest-asyncio` to patch `asyncio`.

    ```python
    import nest_asyncio
    nest_asyncio.apply()
    ```

See the [nest-asyncio](https://pypi.org/project/nest-asyncio/) Python package
for more information.
