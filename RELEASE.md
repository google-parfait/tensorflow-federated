# Release 0.3.0

## Breaking Changes

* Rename tff.learning.federated_average to tff.learning.federated_mean.
* Rename 'func' arguments to 'fn' throughout the API.

## Bug Fixes

* Assorted fixes to typos in documentation and setup scripts.

# Release 0.2.0

## Major Features and Improvements

* Updated to use TensorFlow version 1.13.1.
* Implemented Federated SGD in `tff.learning.build_federated_sgd_process()`.

## Breaking Changes

* `next()` function of `tff.utils.IteratedProcess`s returned by `build_federated_*_process()` no longer unwraps single value tuples (always returns a tuple).

## Bug Fixes

* Modify setup.py to require TensorFlow 1.x and not upgrade to 2.0 alpha.
* Stop unpacking single value tuples in `next()` function of objects returned by `build_federated_*_process()`.
* Clear cached Keras sessions when wrapping Keras models to avoid referencing stale graphs.

# Release 0.1.0

Initial public release.
