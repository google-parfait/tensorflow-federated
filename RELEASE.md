# Release 0.6.0

## Major Features and Improvements

*   Support for multiple outputs and loss functions in `keras` models.
*   Support for stateful broadcast and aggregation functions in federated
    averaging and federated SGD APIs.
*   `tff.utils.update_state` extended to handle more general `state` arguments.
*   Addition of `tff.utils.federated_min` and `tff.utils.federated_max`.
*   Shuffle `client_ids` in `create_tf_dataset_from_all_clients` by default to
    aid optimization.

## Breaking Changes

*   Dependencies added to `requirements.txt`; in particular, `grpcio` and
    `portpicker`.

## Bug Fixes

*   Removes dependency on `tf.data.experimental.NestedStructure`.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Dheeraj R Reddy, @Squadrick.

# Release 0.5.0

## Major Features and Improvements

*   Removed source level TF dependencies and switched from `tensorflow` to
    `tf-nightly` dependency.
*   Add support for `attr` module in TFF type system.
*   Introduced new `tff.framework` interface layer.
*   New AST transformations and optimizations.
*   Preserve Python container usage in `tff.tf_computation`.

## Bug Fixes

*   Updated TFF model to reflect Keras `tf.keras.model.weights` order.
*   Keras model with multiple inputs. #416

# Release 0.4.0

## Major Features and Improvements

*   New `tff.simulation.TransformingClientData` API and associated inifinite
    EMNIST dataset (see http://tensorflow.org/federated/api_docs/python/tff for
    details)

## Breaking Change

*   Normalized `func` to `fn` across the repository (rename some parameters and
    functions)

## Bug Fixes

*   Wrapped Keras models can now be used with
    `tff.learning.build_federated_evaluation`
*   Keras models with non-trainable variables in intermediate layers (e.g.
    BatchNormalization) can be assigned back to Keras models with
    `tff.learning.ModelWeights.assign_weights_to`

# Release 0.3.0

## Breaking Changes

*   Rename `tff.learning.federated_average` to `tff.learning.federated_mean`.
*   Rename 'func' arguments to 'fn' throughout the API.

## Bug Fixes

*   Assorted fixes to typos in documentation and setup scripts.

# Release 0.2.0

## Major Features and Improvements

*   Updated to use TensorFlow version 1.13.1.
*   Implemented Federated SGD in `tff.learning.build_federated_sgd_process()`.

## Breaking Changes

*   `next()` function of `tff.utils.IteratedProcess`s returned by
    `build_federated_*_process()` no longer unwraps single value tuples (always
    returns a tuple).

## Bug Fixes

*   Modify setup.py to require TensorFlow 1.x and not upgrade to 2.0 alpha.
*   Stop unpacking single value tuples in `next()` function of objects returned
    by `build_federated_*_process()`.
*   Clear cached Keras sessions when wrapping Keras models to avoid referencing
    stale graphs.

# Release 0.1.0

Initial public release.
