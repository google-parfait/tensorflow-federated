# Release 0.45.0

## Major Features and Improvements

*   Integrated the `CppToPythonExecutorBridge` into the `CPPExecutorFactory`.
*   Changed Python Remote Executor to decompose and stream structures in Compute
    and CreateValue when _stream_structs is true. Added a bool parameter
    `stream_structs` to
    `tff.backends.native.set_localhost_cpp_execution_context()` API.

## Breaking Changes

*   Renamed `tff.backends.native.set_localhost_cpp_execution_context()` to
    `backends.native.set_sync_local_cpp_execution_context()`.
*   Renamed `tff.framework.ExecutionContext` to
    `tff.framework.SyncExecutionContext` to be consistent with
    `tff.framework.AsyncExecutionContext`.
*   Removed the `SyncSerializeAndExecuteCPPContext` and
    `AsyncSerializeAndExecuteCPPContext` classes.

## Bug Fixes

*   Fixed usage of `typing.Generic` in the learning package.
*   Enabled pytype analysis for some modules.
*   Fixed lint and formatting issues for some modules.

# Release 0.44.0

## Major Features and Improvements

*   Improved the Python type annotations for `tff.program` API.
*   Extended the metrics interface on FunctionalModel to accept the entire
    `BatchOutput` structure from the model `forward_pass` (not just the
    predictions).
*   Introduced a DTensor Executor.

## Bug Fixes

*   Fixed async RuntimeWarning in the `tff.program.NativeFederatedContext`.

# Release 0.43.0

## Major Features and Improvements

*   Improve serialization method to allow structures larger than 2 GiB (~500
    million model parameters):
    *   `tff.learning.models.FunctionalModel`
    *   `tff.programs.FileProgramStateManager`

## Bug Fixes

*   Fix a bug using `copy.deepcopy` for structures of awaitables (non-pickable)
    in `tff.learning.programs`.
*   Fix a bug when resuming an evaluation in
    `tff.learning.programs.EvaluationManager` where the restarted evaluation
    would overwrite released metrics.

# Release 0.42.0

## Major Features and Improvements

*   Reduced memory usage for entropy compression.
*   Updated `com_google_protobuf` version to `v3.19.0`.
*   Removed dependency on `six`.

## Breaking Changes

*   Removed default value for the key parameter from the abstract base class
    `tff.program.ReleaseManager`.

## Bug Fixes

*   Fixed a whitespace syntax issue with shutting down a process when using the
    localhost C++ execution context.
*   Modified `tff.simulation.build_uniform_sampling_fn` so that the output
    raises on non-integer inputs.
*   Only wait a subprocess instance if it is not None.

# Release 0.41.0

## Major Features and Improvements

*   TFF-C++ runtime now installed by default. Note that this release will have a
    significantly larger PIP package size.
*   Introduce `tff.learning.programs` for federated program-logic using the
    `tff.program` APIs.
*   Updated `tensorflow` to version `2.11.0`.
*   Updated `tensorflow_compression` to version `2.11.0`.
*   Updated `bazel_skylib` to version `1.3.0`.

# Release 0.40.0

## Major Features and Improvements

*   Skip model updates that are non-finite in
    `tff.learning.templates.build_apply_optimizer_finalizer`.

## Breaking Changes

*   Removed deprecated APIs in `tff.learning.framework`
*   Update the Python package scripts to use Python 3.10 by default.
*   Remove module wildcard imports from **init**.py files in TFF.
*   Update the Python package scripts to use Python 3.10 by default.

## Bug Fixes

*   Remove `functools.wraps` within `tff.tf_computation`.
*   Fix typo in iNaturalist dataset docstring.

# Release 0.39.0

## Major Features and Improvements

*   Added `tff.learning.models.FunctionModel` support to all methods in
    `tff.learning.algorithms`.
*   Added support for `tf.data.DataSpec` to `tff.types.infer_unplaced_type`.
*   Use a `tensorflow::ThreadPool` for parallelism inside the C++
    `TensorFlowExecutor`.
*   Introduced a new `tff.experimental_tf_fn_computation` tracing decorator that
    uses `FunctionDef` instead of `GraphDef` tracing, providing `tf.function`
    auto-control-dependencies.
*   Renamed `number_of_clients` to `num_clients` in the federated program API.
*   Replaced the following API with composers API in `tff.learning.templates`.
    *   `tff.learning.framework.build_model_delta_optimizer_process`
    *   `tff.learning.framework.ClientDeltaFn`

## Bug Fixes

*   Fixed a bug in the “Client-efficient large-model federated learning”
    tutorial to use the correct dense shape.

# Release 0.38.0

## Major Features and Improvements

*   Added `tff.learning.models.FunctionalModel` support to
    `tff.learning.algorithms.build_mime_lite`.
*   Updated `tensorflow-privacy` to version `0.8.6`.
*   Added an abstract interface describing an asynchronous context
*   Removed references to `tff.framework.Context`.
*   Added `tff.simulation.datasets.gldv2.get_synthetic`.
*   Added prefetching data source in `tff.program.PrefetchingDataSource`.

## Breaking Changes

*   Deleted deprecated
    `tff.learning.framework.build_encoded_broadcast_process_from_model`.
*   Deprecated `tff.learning.ModelWeights` and alias
    `tff.learning.framework.ModelWeights`, has now moved to
    `tff.learning.models.ModelWeights`. Code should be updated before the next
    release.

## Bug Fixes

*   Fixed a bug with variable creation order of metrics in
    `tff.learning.models.functional_model_from_keras`.
*   Improved `tff.tf_computation` tracing to also trace `functools.partial`
    objects.

## Known Bugs

*   Colab compatibility: TFF requires Python 3.9 while Colab runtime uses Python
    3.7.

# Release 0.37.0

## Major Features and Improvements

*   Added support for Python 3.10.
*   Improved support for `numpy` values in the `tff.program` API.
*   Increased dataset serialization size limit to 100MB.
*   Added a new method `tff.learning.ModelWeights.convert_variables_to_arrays`.
*   Added new metrics aggregation factories under `tff.learning.metrics`.
*   Parallelized aggregation in `tff.framework.ComposingExecutorFactory`.

## Breaking Changes

*   Updated to use `jax` and `jaxlib` version `0.3.14`.
*   Renamed `tff.program.CoroValueReference` to
    `tff.program.AwaitableValueReference` to reflect the relaxed contract.

## Bug Fixes

*   Improved documentation for `tff.simulation.build_uniform_sampling_fn`,
    `tff.learning.robust_aggregator`,
    `tff.aggregators.PrivateQuantileEstimationProcess`.
*   Fixed documentation bug for tutorial “High-performance Simulation with
    Kubernetes”.
*   Fixed bug where momentum hyperparameters were added to SGDM optimizer when
    momentum was set to 0.
*   Removed assertion that preprocessed datasets in a
    `tff.simulation.baselines.BaselineTask` have the same element structure.
*   Fixed a memory leak when moving numpy arrays across the Python and C++
    boundary in the C++ executor.
*   Fixed bug in the federated program API when using multiple release managers
    to release the same value.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:
Madhava Jay, nbishdev@

# Release 0.36.0

## Major Features and Improvements

*   Added support for `tff.learning.models.FunctionalModel` to
    `tff.learning.algorithms.build_fed_sgd` and
    `tff.learning.algorithms.build_fed_prox`.
*   Increased the gRPC message limit from 1 GB to 2 GB.
*   Added hyperparameter getters/setters to various components in tff.learning.

## Breaking Changes

*   Updated `tensorflow` to version `2.10`.

## Bug Fixes

*   Improved documentation for
    `tff.analytics.heavy_hitters.iblt.build_iblt_computation()`.
*   Fixed incorrect docstring of `tff.federated_select`.
*   Fixed typo in federated program example.

# Release 0.35.0

## Major Features and Improvements

*   Added get/set_hparams methods to `tff.learning.templates.ClientWorkProcess`.
*   Added `tff.learning.algorithms.build_mime_lite_with_optimizer_schedule`.
*   Updated `tensorflow-privacy` to version `0.8.5`.
*   Added `tff.learning.entropy_compression_aggregator`.
*   Added `tff.aggregators.EliasGammaEncodedSumFactory`.
*   Added `tff.program.ClientIdDataSource` and
    `tff.program.ClientIdDataSourceIterator`, for working with a data source of
    ids of federated clients.

## Breaking Changes

*   Removed prototype IREE backend.
*   Added new dependency on TensorFlow Compression.

## Bug Fixes

*   Fixed implementation of the `loading_remote_data` tutorial.
*   Corrected the docstring of
    `tff.simulation.datasets.stackoverflow.get_synthetic`.

## Known Bugs

*   TFF's Python 3.9 typing conflicts with Colab's Python 3.7 runtime.

# Release 0.34.0

## Major Features and Improvements

*   Updated to use `Bazel` version `5.3.0`.
*   Updated the conventions used to specify the version of a Python dependency,
    see https://github.com/tensorflow/federated/blob/main/requirements.txt for
    more information.
*   Updated the `setup.py` to explicitly fail to `pip install` in Python 3.10.
    This has always been failing at runtime, but now explicitly fails to install
    using `pip`.
*   Refreshed loading_remote_data notebook content and added content for
    `FederatedDataSource`.
*   Added a TFF `type_signature` attribute to objects of type `MapReduceForm`.
*   Added a
    [series](https://github.com/tensorflow/federated/blob/main/docs/design/TFF_101_lingua_federata.pdf)
    [of](https://github.com/tensorflow/federated/blob/main/docs/design/TFF_102_executors.pdf)
    [slides](https://github.com/tensorflow/federated/blob/main/docs/design/TFF_103_transformations.pdf)
    to the GitHub repo (so not part of the PIP package) which detail a technical
    deep dive into TFF.

## Breaking Changes

*   Bumped tf-privacy version to `0.8.4`.
*   Bumped tf-model-optimization version to `0.8.3`.
*   Removed `initialize` from `MapReduceForm`.
*   `SequenceType` now automatically casts any `StructWithPythonType` that
    contains a `list` to a `tuple` for `tf.data` compatibility.
*   Unified the `model_fn` and `model` parameters of
    `tff.learning.algorithms.build_weighted_fed_avg`.
*   `MapReduceForm` now takes a `type_signature` argument in its constructor,
    and no longer takes an `initialize` argument.
*   `MapReduceForm` no longer contains an `initialize` attribute.

## Bug Fixes

*   Relaxed overly strict type equivalence check to assignability in TFF-TF code
    generation.

# Release 0.33.0

## Major Features and Improvements

*   Extend `tff.analytics.heavy_hitters.iblt` with `create_chunker` API for
    encoding non-Unicode strings.
*   Extend `tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation` with
    an optional `record_aggregation_factory` argument.

## Breaking Changes

*   Replaced `ModularClippingSumFactory` with `SecureModularSumFactory` in
    `tff.analytics.build_hierarchical_histogram_process`.

## Known Bugs

*   TFF's python 3.9 typing conflicts with Colab's Python 3.7 support.

# Release 0.32.0

## Major Features and Improvements

*   Add a MimeLite implementation that allows from optimizer learning rate
    scheduling in
    `tff.learning.algorithms.build_mime_lite_with_optimizer_schedule`.

## Breaking Changes

*   None

## Bug Fixes

*   None

## Known Bugs

*   TFF's python 3.9 typing conflicts with Colab's Python 3.7 support.

# Release 0.31.0

## Major Features and Improvements

*   Added `ReleaseManager`s to make authoring program logic more convenient.
*   Updated TFFs `attrs` dependency to version `21.4.0`.
*   Update TFFs `tensorflow-privacy` dependency to version `0.8.1`.

## Breaking Changes

*   Changed `tff.learning.BatchOutput` from an attrs class to a namedtuple.
*   Removed unused `tff.learning.framework.parameter_count_from_model` API.

# Release 0.30.0

## Major Features and Improvements

*   Add tests for `namedtuple`s in the `tff.program` package.
*   Add `num_subrounds` parameter to the mergeable context, allowing callers to
    optionally sequentialize subrounds.
*   Add metrics support to `tff.learning.models.FunctionalModel`, including
    updates to the helper function `create_functional_metric_fns` and the
    downstream caller `tff.learning.algorithms.build_weighted_fed_avg`.

## Bug Fixes

*   Fix typo in the types constructed for testing the `tff.program` package.
*   Fix some program example typos.
*   Fix tests that don't seem to be running under the CI.
*   Fix naming bug for Python mergeable execution.
*   Ensure exceptions raised from remote executor stub implement gRPC error
    interface.
*   Update `tff.structure.Struct` integration with JAX pytrees to not flatten
    the entire structure.
*   Use Python 3.7 compatible type annotations until Colab updates to Python
    3.9.

# Release 0.29.0

## Major Features and Improvements

*   Update the `MemoryReleaseManager` to save `type_signature` when releasing
    values.
*   Add a `type_signature` parameter to the `ReleaseManager.release` method.
*   Unify retryability logic between TFF-C++ and TFF-Python.
*   Update the TFF contributions and collaboration links to point to the Discord
    server.

## Breaking Changes

*   Move Python executor stacks file to `python_executor_stacks.py` in
    `executor_stacks` directory.

## Bug Fixes

*   Ensure that dataset deserialization will return ragged and sparse tensors,
    as needed according to the TFF type of the dataset.
*   Make `metric_finalizers` use metric constructors if available.

# Release 0.28.0

## Major Features and Improvements

*   Updated tutorials to use `tff.learning.algorithms` API.
*   Asynchronous TFF execution contexts no longer assume a single global
    cardinality; they concurrently invoke any computation for which concurrency
    is requested.

## Breaking Changes

*   Removed `tff.learning.build_federated_averaging_process`; users should
    migrate to `tff.learning.algorithms.build_weighted_fed_avg`.

## Bug Fixes

*   Clarified semantics for TFF-C++ multimachine `Dispose`, `DisposeExecutor`,
    and executor keying, to avoid raising exceptions and spamming logs in the
    course of normal operation.
*   Fixed unsigned integer overflow for TFF-C++
    `max_concurrent_computation_calls`.
*   Normalizes on call-dominant form before attempting to compile to
    `MergeableCompForm`, removing spurious failures on dependent-aggregation
    checking.

## Known Bugs

*   Serialization / deserialization of tf.data.Datasets yielding non-dense
    tensors for multimachine runtime may encounter issues:
    *   `tff.framework.deserialize_value` may fail to deserialize
        tf.data.Datasets yielding RaggedTensors or SparseTensors.
    *   `tff.framework.serialize_value` may fail to serialize tf.data.Datasets
        yielding SparseTensors.

# Release 0.27.0

## Major Features and Improvements

*   New Colab notebook illustrating how to use `DataBackend` to load remote
    datasets.
*   Added a CreateDataDescriptor helper function.
*   Added a worker binary serving the TFF-C++ executor service.

## Bug Fixes

*   Fixed bug with intermediate aggregation and controller failures, causing
    hangs.

# Release 0.26.0

## Major Features and Improvements

*   Updated TensorFlow to `2.9.1`.
*   Update pybind11 to `2.9.2`.
*   Re-enable cpp_fast_protos.
*   Introduces container class to run coroutines in a dedicated thread, allowing
    TFF’s synchronous execution interfaces to be used in conjunction with other
    asyncio code.
*   Use latest TFF version in Colab notebook links.
*   Rename the helper functions that create test `MeasuredProcess`es.
*   Add a compiler transform checking Tensorflow computations against list of
    allowed ops.
*   Explicitly specify return types in the `program` package.
*   Adds convenience function for setting a local async CPP execution context.
*   Move jax components into a non-experimental namespace.

## Breaking Changes

*   Switch compilation flag `_GLIBCXX_USE_CXX11_ABI` to `1`.

# Release 0.25.0

## Major Features and Improvements

*   Adds error message logging to TFF C++ execution context.
*   Adds test coverage for C++ runtime with aggregators.
*   Redefines 'workers going down with fixed clients per round' test.
*   Add complete examples of using `DataBackend` with TFF comps.
*   Updated the MapReduceForm documentation to include the two additional secure
    sum intrinsics.
*   tff.learning
    *   Relax the type check on LearningProcess from strictly SequenceType to
        also allow structures of SequenceType.

## Breaking Changes

*   Remove usage of `tff.test.TestCase`, `tff.test.main()`, and delete
    `test_case` module.
*   Update test utility docstrings to use consistent vocabulary.
*   Update to TensorFlow 2.9.0
*   Rename up `compiler/test_utils` to `compiler/building_block_test_utils`.
*   Remove some unnecessary usage of `pytype: skip-file`.
*   Specify the `None` return type of `ReleaseManager.release`.
*   Remove usage of deprecated numpy types.
*   Replace depreciated `random_integers` with `randint`.

## Bug Fixes

*   Fix numpy warning.

# Release 0.24.0

## Major Features and Improvements

*   Added `asyncio.run` call to metrics manager release calls to ensure
    compatibility with
    https://github.com/tensorflow/federated/commit/a98b5ed6894c536549da06b4cc7ed116105dfe65.
*   Added an example and documentation for the Federated Program API.
*   Improved `model_update_aggregator` to support structures with mixed floating
    dtypes.
*   Create a mirror of
    `tff.simulation.compose_dataset_computation_with_iterative_process` for
    `tff.learning.templates.LearningProcess`.
*   Added logging of expected sequential computations to local TFF-C++ runtime.

## Breaking Changes

*   Moved asserts from `tff.test.TestCase` to `tff.test.*` as functions.
*   Removed `assert_type_assignable_from` function.
*   Moved `assert_nested_struct_eq` to the `type_conversions_test` module.
*   Removed `client_train_process` and fedavg_ds_loop comparisons.

## Bug Fixes

*   Fixed comparisons to enums in the benchmarks package.
*   Fixed `async_utils.SharedAwaitable` exception raiser.
*   Fixed various lint errors.

# Release 0.23.0

## Major Features and Improvements

*   Deprecated `tff.learning.build_federated_averaging_process`.
*   Added an API to convert `tf.keras.metrics.Metric` to a set of pure
    `tf.functions`.

## Breaking Changes

*   Renamed `ProgramStateManager.version` to `ProgramStateManager.get_versions`.

## Bug Fixes

*   Fixed the "datasets/" path in the working with TFF's ClientData tutorial.

# Release 0.22.0

## Major Features and Improvements

*   Updated .bazelversion to `5.1.1`.
*   Updated the `tff.program` API to use `asyncio`.
*   Exposed new APIs in the `tff.framework` package:
    *   `tff.framework.CardinalitiesType`.
    *   `tff.framework.PlacementLiteral`.
    *   `tff.framework.merge_cardinalities`.
*   `tff.analytics`
    *   Added new `analytic_gauss_stddev` API.

## Breaking Changes

*   Renamed `ProgramStateManager.version` to `ProgramStateManager.get_versions`.

## Bug Fixes

*   Fixed some Python lint errors related to linting Python 3.9.
*   Cleaned up stale TODOs throughout the codebase.

## Known Bugs

*   Version 0.21.0 currently fails to import in
    [colab](https://colab.research.google.com) if the version of Python is less
    than Python 3.9. Please use a runtime with a version of Python greater than
    Python 3.9 or use TFF version 0.20.0.

# Release 0.21.0

## Major Features and Improvements

*   `tff.analytics`
    *   Added new `tff.analytics.IbltFactory` aggregation factory.
    *   Added new IBTL tensor encoder/decoder libraries and uses them in
        `tff.analytics.heavy_hitters.iblt.build_iblt_computation`.
*   `tff.aggregator`
    *   Added `as_weighted_aggregator` to the `tff.aggregator.Factory` API.
*   `tff.learning`
    *   Improved compilation and execution performance of
        `tff.learning.metrics.secure_sum_then_finalize` by grouping tensors by
        DType.
    *   Added `set_model_weights` method and default implementation to
        `tff.learning.templates.LearningProcess`.
    *   Added a new `reset_metrics` attribute to `tff.learning.Model`.
    *   Added `schedule_learning_rate` to `tff.learning.optimizers`.
    *   Added new `tff.learning.ddp_secure_aggregator` for Distributed
        Differential Privacy.
*   `tff.simulation`
    *   Added an option to distort train images in the CIFAR-100 baseline task.
    *   Changed the default sequence length for the Shakespeare baseline task to
        a more reasonable value.
*   Core
    *   Switched runtime to create new RemoteExecutors with different
        cardinalities, rather than resetting the cardinality in the remote
        service.

## Breaking Changes

*   Removed support for Python 3.7 and 3.8, TFF supports 3.9 and later.
*   Removed deprecated attributes `report_local_outputs` and
    `federated_output_computation` from `tff.learning.Model`
*   Removed the `ingest` method from `tff.Context`

## Bug Fixes

*   Multiple typos in tests, code comments, and pydoc.

## Known Bugs

*   Sequences (datasets) of SparseTensors don't work on the C++ runtime.
*   Computations when `CLIENTS` cardinality is zero doesn't work on the Python
    runtime.
*   Assigning variables to a Keras model after construction inside a `model_fn`
    results in a non-deterministic graph.

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
