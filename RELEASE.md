# Release 0.4.0

## Major Features and Improvements

* New `tff.simulation.TransformingClientData` API and associated inifinite EMNIST dataset (see tensorflow.org/federated/api\_docs/python/tff for details)

## Breaking Change

* Normalized `func` to `fn` across the repository (rename some parameters and functions)

## Bug Fixes

* Wrapped Keras models can now be used with `tff.learning.build_federated_evaluation`
* Keras models with non-trainable variables in intermediate layers (e.g.  BatchNormalization) can be assigned back to Keras models with `tff.learning.ModelWeights.assign_weights_to`

# Release 0.3.0

## Breaking Changes

* Rename tff.learning.federated\_average to tff.learning.federated\_mean.
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
