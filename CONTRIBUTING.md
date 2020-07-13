# Contributing

## Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a
couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement
(CLA).

*   If you are an individual writing original source code and you're sure you
    own the intellectual property, then you'll need to sign an
    [individual CLA](http://code.google.com/legal/individual-cla-v1.0.html).
*   If you work for a company that wants to allow you to contribute your work,
    then you'll need to sign a
    [corporate CLA](http://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and
instructions for how to sign and return it. Once we receive it, we'll be able to
accept your pull requests.

Note: Only original source code from you and other people that have signed the
CLA can be accepted into the main repository.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

## Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Code Style, Guidelines, and Best Practices

### General Guidelines

*   Python code should adhere to the
    [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).

*   Fromat your changes.

    *   Install [yapf](https://pypi.org/project/yapf/).

    *   Format unstaged changes, to Python files, in place.

    ```shell
    git diff --name-only \
      | sed '/.*\.py/!d' \
      | xargs yapf --in-place
    ```

*   Lint your changes.

    *   Install [pylint](https://pypi.org/project/pylint/).

    *   Lint unstaged changes to Python files.

    ```shell
    git diff --name-only \
      | sed '/.*\.py/!d' \
      | xargs pylint
    ```

*   Python code must support Python 3 usage.

*   Include a license at the top of new files.

    *   [Python license example](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/__init__.py#L1)

### TensorFlow-specific Guidelines.

*   TensorFlow code should follow the
    [TensorFlow Style Guide](https://www.tensorflow.org/community/style_guide).

*   TensorFlow code used with TFF should support both "graph mode" and "eager
    mode" execution. Good eager-mode design principles should be followed,
    including:

    *   Avoid saving references to tensors where the value may change.
    *   All TensorFlow functions should work correctly when annotated with
        `tf.function`. Such functions should only return multiple outputs (e.g.,
        as a tuple) if it always makes sense to compute all of these values at
        the same time. The exception is Variable creation, which should always
        happen outside of @tf.function decorated functions.
    *   Collections should not be used, unless it is unavoidable to support TF
        1.0.
    *   State such as `tf.Variable`s should be tracked (only) by keeping a
        reference to the Python Variable object.
    *   Use program-order-semantics in `tf.function`s rather than explicit
        control dependencies when possible. If line of code A should execute
        before line B, then the lines should occur in that order.
    *   Don't write TF code which can only be correctly called once.

*   **dict vs OrderedDict**: Prefer `OrderedDict`. The names of `tf.Variable`s
    may depend on the order in which they are created, due to name
    uniquification. Since `dict`s have arbitrary iteration order, this
    non-determinism can lead to Checkpoint-incompatible graphs. Furthermore, TFF
    type signatures constructed from unordered dictionaries may also mismatch as
    their entries are permuted.

### TFF-Learning-specific Guidelines

*   While not a requirement, in our examples we use the conventions that metrics
    and counters associated with a `tff.learning.Model` are named using
    `lower_with_under` style, as with Python identifiers.
