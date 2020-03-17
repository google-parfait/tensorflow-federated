# Release 0.13.1

## Bug Fixes

*   Fixed issues in tutorial notebooks.

# Release 0.13.0

## Major Features and Improvements

*   Updated `absl-py` package dependency to `0.9.0`.
*   Updated `h5py` package dependency to `2.8.0`.
*   Updated `numpy` package dependency to `1.17.5`.
*   Updated `tensorflow-privacy` package dependency to `0.2.2`.

## Breaking Changes

*   Deprecated `dummy_batch` parameter of the `tff.learning.from_keras_model`
    function.

## Bug Fixes

*   Fixed issues with executor service using old executor API.
*   Fixed issues with remote executor test using old executor API.
*   Fixed issues in tutorial notebooks.

# Release 0.12.0

## Major Features and Improvements

*   Upgraded tensorflow dependency from `2.0.0` to `2.1.0`.
*   Upgraded tensorflow-addons dependency from `0.6.0` to `0.7.0`.
*   Upgraded attr dependency from `18.2` to `19.3`.
*   Upgraded tfmot dependency from `0.1.3` to `0.2.1`.
*   Added a federated partition of the CIFAR-100 dataset to
    `tff.simulation.datasets.cifar100`.
*   Made the high performance, parallel executor the default (replacing the
    reference executor).
*   Added a new `tff.learning.build_personalization_eval` for evaluating model
    personalization strategies.
*   Added new federated intrinsic `tff.federated_secure_sum`.
*   `tff.learning.build_federated_averaing_process()` now takes a
    `client_optimizer_fn` and a `tff.learning.Model`.
    `tff.learning.TrainableModel` is now deprecated.
*   Improved performance in the high performance executor stack.
*   Implemented and exposed `tff.framework.ExecutorFactory`; all
    `tff.framework...executor_factory` calls now return an instance of this
    class.
*   Added `remote_executor_example` binary which demonstrates using the
    RemoteExecutor across multi-machine deployments.
*   Added `close()` method to the Executor, allowing subclasses to proactively
    release resources.
*   Updated documentation and scripts for creating Docker images of the TFF
    runtime.
*   Automatically call `tff.federated_zip` on inputs to other federated
    intrinsics.

## Breaking Changes

*   Dropped support for Python2.
*   Renamed `tff.framework.create_local_executor` (and similar methods) to
    `tff.framework.local_executor_factory`.
*   Deprecated `federated_apply()`, instead use `federated_map()` for all
    placements.

## Bug Fixes

*   Fixed problem with different instances of the same model having different
    named types. `tff.learning.ModelWeights` no longer names the tuple fields
    returned for model weights, instead relying on an ordered list.
*   `tff.sequence_*` on unplaced types now correctly returns a `tff.Value`.

## Known Bugs

*   `tff.sequence_*`.. operations are not implemented yet on the new
    high-performance executor stack.
*   A subset of previously-allowed lambda captures are no longer supported on
    the new execution stack.

# Release 0.11.0

## Major Features and Improvements

*   Python 2 support is now deprecated and will be removed in a future release.
*   `federated_map` now works with both `tff.SERVER` and `tff.CLIENT`
    placements.
*   `federated_zip` received significant performance improvements and now works
    recursively.
*   Added retry logic to gRPC calls in the execution stack.

## Breaking Changes

*   `collections.OrderedDict` is now required in many places rather than
    standard Python dictionaries.

## Bug Fixes

*   Fixed computation of the number of examples when Keras is using multiple
    inputs.
*   Fixed an assumption that `tff.framework.Tuple` is returned from
    `IterativeProcess.next`.
*   Fixed argument packing in polymorphic invocations on the new executor API.
*   Fixed support for `dir()` in `ValueImpl`.
*   Fixed a number of issues in the Colab / Jupyter notebook tutorials.

# Release 0.10.1

## Bug Fixes

*   Updated to use `grpcio` `1.24.3`.

# Release 0.10.0

## Major Features and Improvements

*   Add a `federated_sample` aggregation that is used to collect a sample of
    client values on the server using reservoir sampling.
*   Updated to use `tensorflow` `2.0.0` and `tensorflow-addons` `0.6.0` instead
    of the coorisponding nightly package in the `setup.py` for releasing TFF
    Python packages.
*   Updated to use `tensorflow-privacy` `0.2.0`.
*   Added support for `attr.s` classes type annotations.
*   Updated streaming `Execute` method on `tff.framework.ExecutorService` to be
    asynchronous.
*   PY2 and PY3 compatability.

# Release 0.9.0

## Major Features and Improvements

*   TFF is now fully compatible and dependent on TensorFlow 2.0
*   Add stateful aggregation with differential privacy using TensorFlow Privacy
    (https://pypi.org/project/tensorflow-privacy/).
*   Additional stateful aggregation lwith compression using TensorFlow Model
    Optimization (https://pypi.org/project/tensorflow-model-optimization/).
*   Improved executor stack for simulations, documentation and scripts for
    starting simulations on GCP.
*   New libraries for creating synthetic IID and non-IID datsets in simulation.

## Breaking Changes

*   `examples` package split to `simulation` and `research`.

## Bug Fixes

*   Various error message string improvements.
*   Dataset serialization fixed for V1/V2 datasets.
*   `tff.federated_aggregate` supports `accumulate`, `merge` and `report`
    methods with signatures containing tensors with undefined dimensions.

# Release 0.8.0

## Major Features and Improvements

*   Improvements in the executor stack: caching, deduplication, bi-directional
    streaming mode, ability to specify physical devices.
*   Components for integration with custom mapreduce backends
    (`tff.backends.mapreduce`).
*   Improvements in simulation dataset APIs: ConcreteClientData, random seeds,
    stack overflow dataset, updated documentation.
*   Utilities for encoding and various flavors of aggregation.

## Breaking Changes

*   Removed support for the deprecated `tf.data.Dataset` string iterator handle.
*   Bumps the required versions of grpcio and tf-nightly.

## Bug Fixes

*   Fixes in notebooks, typos, etc.
*   Assorted fixes to align with TF 2.0.
*   Fixes thread cleanup on process exit in the high-performance executor.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Gui-U@, Krishna Pillutla, Sergii Khomenko.

# Release 0.7.0

## Major Features and Improvements

*   High-performance simulation components and tutorials.

## Breaking Changes

*   Refactoring/consolidation in utility functions in tff.framework.
*   Switches some of the tutorials to new PY3-only executor stack components.

## Bug Fixes

*   Includes the `examples` directory in the pip package.
*   Pip installs for TensorFlow and TFF in turorials.
*   Patches for asyncio in tutorials for use in Jupyter notebooks.
*   Python 3 compatibility issues.
*   Support for `federated_map_all_equal` in the reference executor.
*   Adds missing implementations of generic constants and operator intrinsics.
*   Fixes missed link in compatibility section of readme.
*   Adds some of the missing intrinsic reductions.

## Thanks to our Contributors

This release contains contributions from many people at Google.

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
