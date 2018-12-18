# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

## Code Style, Guidelines, and Best Practices

### General Guidelines

*   Python code should adhere to the
    [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).

*   Python code must support Python2 and Python3 usage.

*   Generally, function names should be verbs, e.g. `to_var_dict` rather than
    `var_dict`.

### TensorFlow-specific Guidelines.

*   TensorFlow code should follow the
    [TensorFlow Style Guide](https://www.tensorflow.org/community/style_guide).

*   TensorFlow code used with TFF should support both "graph mode" and "eager
    mode" execution. Good eager-mode design principles should be followed,
    including:

    *   Avoid saving references to tensors where the value may change.
    *   All TensorFlow functions should work correctly when annotated with
        `tf.function` or `tf.contrib.eager.defun`. Such functions should only
        return multiple outputs (e.g., as a tuple) if it always makes sense to
        compute all of these values at the same time.
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
