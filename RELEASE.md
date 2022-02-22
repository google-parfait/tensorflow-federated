# Release 0.20.0

## Major Features and Improvements

*   Added `tff.program` API; this API is still in active development but can be
    used to compose shared and platform specific: program logic, components, and
    privacy concepts to create federated programs.
*   Added support for Python 3.9.
*   Added CelebA and iNaturalist datasets to `tff.simulation.datasets`.
*   Added `tff.analytics` API for federated analytics, including private heavy
    hitters algorithms.
*   Added `tff.learning.algorithms` API, including TFF implementations of
    FedProx, FedAvg with learning rate scheduling, federated k-Means, and
    MimeLite.
*   Added `tff.learning.metrics` API to support easy configuration of
    cross-client metrics aggregation via the new `metrics_aggregator` argument.
*   Added `metrics_aggregator` argument to
    `tff.learning.build_federated_averaging_process` and
    `tff.learning.build_federated_evaluation`.
*   Added `report_local_unfinalized_metrics` and `metric_finalizers` methods to
    `tff.learning.Model` and deprecated `report_local_outputs` and
    `federated_output_computation`.
*   Added `tff.learning.optimizers` API for building purely functional
    optimizers and implementations of SGD, Adagrad, Rmsprop, Adam, Yogi,
*   Added `tff.learning.reconstruction` API for building partially local
    federated learning algorithms, including Federated Reconstruction.
*   Added `tff.learning.templates` API to support building learning algorithms
    in a modular fashion.
*   Added `tff.simulation.baselines` API to support evaluating learning
    algorithms on a suite of representative tasks.
*   Added `tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation` to
    support the [DP-FTRL algorithm](https://arxiv.org/abs/2103.00039).
*   Added `tff.aggrgators.SecureModularSumFactory`
*   Added `tff.aggregators.DiscreteFourierTransformFactory` and
    `tff.aggregators.HadamardTransformFactory` to support rotation-based
    aggregators.
*   Added `tff.aggregators.concat_factory` for aggregating structures as a
    single tensor.
*   Added `tff.backends.native.create_mergeable_comp_execution_context`,
    `tff.backends.native.set_mergeable_comp_execution_context`; these can be
    used with a distributed runtime to scale to tens of thousands of clients.
*   Improved performance of many `tff.simulation.datasets.ClientData`
    subclasses.
*   Added `tff.simulation.datasets.ClientData.serializable_dataset_fn`
    attribute, enabling dataset creation within TF/TFF computations.
*   Added `debug_measurements` option to aggregators in `tff.learning`.
*   Added support for unambiguous zero-client aggregations.
*   Added support for Python dataclasses as function parameters and return
    values for TFF computations.
*   Added automatic insertion of `tff.federated_zip` to invocation of
    user-defined TFF federated computations.
*   Added utilities to `tff.simulation.datasets` for saving federated datasets
    to a SQL database compatible with `tff.simulation.datasets.SqlClientData`.
*   Added `tff.learning.models.FunctionalModel` and
    `tff.learning.models.functional_model_from_keras`.
*   Increased max flow of tensors. Tensors now flow here, there, and everywhere.
*   Updated the Python dependencies:
*   Updated `absl-py` to version `1.0.0`.
*   Updated `attrs` to version `21.2.0`.
*   Added `farmhashpy` version `0.4.0`.
*   Updated `jax` to version `0.2.27`.
*   Updated `jaxlib` to version `0.1.76`.
*   Updated `numpy` to version `1.21.4`.
*   Removed `retrying`.
*   Updated `tensorflow-model-optimization` to version `0.7.1`.
*   Updated `tensorflow-model-optimization` to version `0.7.3`.
*   Updated `tensorflow` to version `2.8.0`.
*   Added support for building many dependencies including `tensorflow` using
    Bazel.
*   Updated the Bazel dependencies:
*   Updated `rules_python` to version `0.5.0`.
*   Updated `com_google_protobuf` to version `v3.18.0-rc1`.
*   Added `absl_py` version `1.0.0`.
*   Added `com_google_googletest` version `release-1.11.0`.
*   Added `io_bazel_rules_go` version `v0.29.0`.
*   Added `bazel_skylib` version `1.0.3`.
*   Added `pybind11_abseil`.
*   Added `pybind11_bazel`.
*   Added `pybind11_protobuf`.
*   Added `com_google_absl` version `20211102.0`.
*   Added `tensorflow_org` version `v2.8.0`.

## Breaking Changes

*   Removed support for building source on MacOS.
*   Removed support for Python 3.6.
*   Removed symbol `tff.framework.type_contains`, use `tff.types.contains`
    instead.
*   Removed many symbols from `tff.simulation`, these can be found in
    `tff.program` instead.
*   Removed support for converting non-OrderedDict mapping types to
    `tff.Value`s.
*   Removed `tff.simulation.datasets.ClientData.from_clients_and_fn` in favor of
    `tff.simulation.datasets.ClientData.from_clients_and_tf_fn`.
*   Restricted `tff.simulation.datasets.ClientData.preprocess` to only support
    TF-serializable functions.
*   Removed `tff.backends.reference`, and the reference context it contained.
*   Removed `tff.learning.build_federated_sgd_process` in favor of
    `tff.learning.algorithms.build_fed_sgd`.
*   Removed `tff.simulation.run_simulation` in favor of
    `tff.simulation.run_training_process`.
*   Removed `tff.learning.framework.EnhancedModel`.
*   Removed `tff.learning.framework.build_stateless_mean`.

## Bug Fixes

*   Fixed broken links in documentation.
*   Fixed many pytype errors.
*   Fixed some inconsistencies in Bazel visibility.
*   Fixed bug where `tff.simulation.datasets.gldv2.load_data()` would result in
    an error.

# Release 0.19.0

## Major Features and Improvements

*   Introduced new intrinsics: `federated_select` and `federated_secure_select`.
*   New `tff.structure_from_tensor_type_tree` to help manipulate structures of
    `tff.TensorType` into structures of values.
*   Many new `tff.aggregators` factory implementations.
*   Introduced `tf.data` concept for data URIs.
*   New `tff.type` package with utilities for working with `tff.Type` values.
*   Initial experimental support for `tff.jax_computation`.
*   Extend `tff.tf_computation` support to `SpareTensor` and `RaggedTensor`.

## Breaking Changes

*   Update gRPC dependency to 1.34.
*   Moved `ClientData` interface and implementations to
    `tff.simulation.datasets`.
*   Renamed `tff.utils.update_state` to `tff.structure.update_struct`.
*   Removed the `tff.utils` namespace, all symbols have migrated, many to
    `tff.aggregators`.
*   Moved infinite EMNIST dataset to federated research repository.
*   Removes `rpc_mode` argument to remote executors, along with streaming mode.
*   Removes deprecated `tff.federated_apply`.
*   Removes `tff.federated_reduce`, all usages can use
    `tff.federated_aggregate`.
*   Removes `HDF5ClientData` and `h5py` pip dependency.
*   Removes `setattr` functionality on `tff.ValueImpl`.

## Bug Fixes

*   Improved `tf.GraphDef` comparisons.
*   Force close generators used for sending functions to computation wrappers,
    avoiding race conditions in Colab.
*   Fix tracing libraries asyncio usage to be Python3.9 compatible.
*   Fix issue with destruction of type intern pool destructing and `abc`.
*   Fix type interning for tensors with unknown dimensions.
*   Fix `ClientData.create_dataset_from_all_clients` consuming unreasonable
    amounts of memory/compute time.

# Release 0.18.0

## Major Features and Improvements

*   Extended the `tff.simulation` package to add many new tools for running
    simulations (checkpoints and metrics managers, client sampling functions).
*   Extended the `tff.aggregators` package with a number of new aggregation
    factories.
*   Added the `tff.structure` API to expose the `Struct` class and related
    functions.
*   Added the `tff.profiler` API to expose useful profiling related functions.
*   Added the `tff.backends.test` package to expose backends that focused on
    testing specifically a way to test a computation with a
    `federated_secure_sum` intrinsic.
*   Added the `tff.experimental` package to expose less stable API.

## Breaking Changes

*   Replaced the `tff.aggregators.AggregationProcessFactory` abstract base class
    with the `tff.aggregators.UnweightedAggregationFactory` and the
    `tff.aggregators.WeightedAggregationFactory` classes.
*   Replaced the `tff.aggregators.ZeroingFactory` class with a
    `tff.aggregators.zeroing_factory` function with the same input arguments.
*   Replaced the `tff.aggregators.ClippingFactory` class with a
    `tff.aggregators.clipping_factory` function with the same input arguments.
*   Updated `tensorflow` package dependency to `2.4.0`.
*   Updated `absl-py` package dependency to `0.10`.
*   Updated `grpcio` package dependency to `1.32.0`.
*   Added a `jaxlib` package dependency at `0.1.55`.
*   Updated `numpy` package dependency to `1.19.2`.
*   Updated `tensorflow-addons` package dependency to `0.12.0`.
*   Updated `tensorflow-model-optimization` package dependency to `0.5.0`.

## Bug Fixes

*   Fixed issue with the `sequence_reduce` intrinsic handling federated types.

# Release 0.17.0

## Major Features and Improvements

*   New `tff.aggregators` package with interfaces for stateful aggregation
    compositions.
*   New Google Landmark Dataset `tff.simulations.dataset.gldv2`
*   New convenience APIs `tff.type_clients` and `tff.type_at_server`
*   Invert control of computation tracing methods to produce clearer Python
    stack traces on error.
*   Move executor creation to a factory pattern in executor service, allowing
    distributed runtimes to be agnostic to number of clients.
*   Significant improvements of type serialization/deserialization
*   New `tff.simulations.compose_dataset_computation_with_iterative_process` API
    to move execution of client dataset construction to executor stack leaves.
*   Extend parameterization of `tff.learning.build_federated_averaging_process`
    with `use_experimental_simulation_loop` argument to better utilize multi-GPU
    setups.

## Breaking Changes

*   Removed `tff.utils.StatefulFn`, replaced by `tff.templates.MeasuredProcess`.
*   Removed `tff.learning.assign_weights_to_keras_model`
*   Stop removing `OptimizeDataset` ops from `tff.tf_computation`s.
*   The `research/` directory has been moved to
    http://github.com/google-research/federated.
*   Updates to `input_spec` argument for `tff.learning.from_keras_model`.
*   Updated TensorFlow dependency to `2.3.0`.
*   Updated TensorFlow Model Optimization dependency to `0.4.0`.

## Bug Fixes

*   Fixed streaming mode hang in remote executor.
*   Wrap `collections.namedtuple._asdict` calls in `collections.OrderedDict` to
    support Python 3.8.
*   Correctly serialize/deserialize `tff.TensorType` with unknown shapes.
*   Cleanup TF lookup HashTable resources in TFF execution.
*   Fix bug in Shakespeare dataset where OOV and last vocab character were the
    same.
*   Fix TFF ingestion of Keras models with shared embeddings.
*   Closed hole in compilation to CanonicalForm.

## Known Bugs

*   "Federated Learning for Image Classification" tutorial fails to load
    `projector` plugin for tensorboard.
    (https://github.com/tensorflow/federated/issues/914)
*   Certain Keras models with activity regularization fail in execution with
    unliftable error (https://github.com/tensorflow/federated/issues/913).

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

amitport, ronaldseoh

# Release 0.16.1

## Bug Fixes

*   Fixed issue preventing Python `list`s from being `all_equal` values.

# Release 0.16.0

## Major Features and Improvements

*   Mirrored user-provided types and minimize usage of `AnonymousTuple`.

## Breaking Changes

*   Renamed `AnonymousTuple` to `Struct`.

# Release 0.15.0

## Major Features and Improvements

*   Updated `tensorflow-addons` package dependency to `0.9.0`.
*   Added API to expose the native backend more conveniently. See
    `tff.backends.native.*` for more information.
*   Added a compiler argument to the `tff.framework.ExecutionContext` API and
    provided a compiler for the native execution environment, which improves
    TFF’s default concurrency pattern.
*   Introduced a new `tff.templates.MeasuredProcess` concept, a specialization
    of `tff.templates.IterativeProcess`.
*   Extends `tff.learning` interfaces to accept `tff.templates.MeasuredProcess`
    objects for aggregation and broadcast computations.
*   Introduce new convenience method `tff.learning.weights_type_from_model`.
*   Introduced the concept of a `tff.framework.FederatingStrategy`, which
    parameterizes the `tff.framework.FederatingExecutor` so that the
    implementation of a specific intrinsic is easier to provide.
*   Reduced duplication in TFF’s generated ASTs.
*   Enabled usage of GPUs on remote workers.
*   Documentation improvements.

## Breaking Changes

*   The `IterativeProcess` return from
    `tff.learning.build_federated_averaging_process` and
    `tff.learning.build_federated_sgd_process` now zip the second tuple output
    (the metrics) to change the result from a structure of federated values to
    to a federated structure of values.
*   Removed `tff.framework.set_default_executor` function, instead you should
    use the more convenient `tff.backends.native.set_local_execution_context`
    function or manually construct a context an set it using
    `tff.framework.set_default_context`.
*   The `tff.Computation` base class now contains an abstract `__hash__` method,
    to ensure compilation results can be cached. Any custom implementations of
    this interface should be updated accordingly.

## Bug Fixes

*   Fixed issue for missing variable initialization for variables explicitly not
    added to any collections.
*   Fixed issue where table initializers were not run if the
    `tff.tf_computation` decorated function used no variables.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

jvmcns@

# Release 0.14.0

## Major Features and Improvements

*   Multiple TFF execution speedups.
*   New `tff.templates.MeasuredProcess` specialization of `IterativeProcess`.
*   Increased optimization of the `tff.templates.IterativeProcess` ->
    `tff.backends.mapreduce.CanonicalForm` compiler.

## Breaking Changes

*   Moved `tff.utils.IterativeProcess` to `tff.templates.IterativeProcess`.
*   Removed `tff.learning.TrainableModel`, client optimizers are now arguments
    on the `tff.learning.build_federated_averaging_process`.
*   Bump required version of pip packages for tensorflow (2.2), numpy (1.18),
    pandas (0.24), grpcio (1.29).

## Bug Fixes

*   Issue with GPUs in multimachine simulations not being utilized, and bug on
    deserializing datasets with GPU-backed runtime.
*   TensorFlow lookup table initialization failures.

## Known Bugs

*   In some situations, TF will attempt to push Datasets inside of tf.functions
    over GPU device boundaries, which fails by default. This can be hit by
    certain usages of TFF,
    [e.g. as tracked here](https://github.com/tensorflow/federated/issues/832).

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

jvmcns@

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
