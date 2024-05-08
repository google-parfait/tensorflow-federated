load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":templates_packages",
        "//tensorflow_federated/python/learning:learning_users",
        "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
        "//tensorflow_federated/python/learning/programs:programs_packages",
    ],
)

package_group(
    name = "templates_packages",
    packages = ["//tensorflow_federated/python/learning/templates/..."],
)

licenses(["notice"])

py_library(
    name = "templates",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":apply_optimizer_finalizer",
        ":client_works",
        ":composers",
        ":distributors",
        ":finalizers",
        ":learning_process",
        ":model_delta_client_work",
    ],
)

py_library(
    name = "apply_optimizer_finalizer",
    srcs = ["apply_optimizer_finalizer.py"],
    deps = [
        ":finalizers",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/optimizers:keras_optimizer",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "apply_optimizer_finalizer_test",
    srcs = ["apply_optimizer_finalizer_test.py"],
    deps = [
        ":apply_optimizer_finalizer",
        ":finalizers",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "client_works",
    srcs = ["client_works.py"],
    deps = [
        ":hparams_base",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "client_works_test",
    size = "small",
    srcs = ["client_works_test.py"],
    deps = [
        ":client_works",
        ":hparams_base",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "composers",
    srcs = ["composers.py"],
    deps = [
        ":apply_optimizer_finalizer",
        ":client_works",
        ":distributors",
        ":finalizers",
        ":learning_process",
        ":model_delta_client_work",
        ":type_checks",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
    ],
)

py_test(
    name = "composers_test",
    size = "small",
    timeout = "moderate",
    srcs = ["composers_test.py"],
    deps = [
        ":apply_optimizer_finalizer",
        ":client_works",
        ":composers",
        ":distributors",
        ":finalizers",
        ":learning_process",
        ":model_delta_client_work",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:model_examples",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "distributors",
    srcs = ["distributors.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "distributors_test",
    size = "small",
    srcs = ["distributors_test.py"],
    deps = [
        ":distributors",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "finalizers",
    srcs = ["finalizers.py"],
    deps = [
        ":hparams_base",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "finalizers_test",
    size = "small",
    srcs = ["finalizers_test.py"],
    deps = [
        ":finalizers",
        ":hparams_base",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "hparams_base",
    srcs = ["hparams_base.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_test(
    name = "hparams_base_test",
    srcs = ["hparams_base_test.py"],
    deps = [
        ":hparams_base",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "learning_process",
    srcs = ["learning_process.py"],
    deps = [
        ":hparams_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:iterative_process",
    ],
)

py_test(
    name = "learning_process_test",
    size = "small",
    srcs = ["learning_process_test.py"],
    deps = [
        ":learning_process",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "model_delta_client_work",
    srcs = ["model_delta_client_work.py"],
    deps = [
        ":client_works",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:dataset_reduce",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/metrics:types",
        "//tensorflow_federated/python/learning/models:functional",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "model_delta_client_work_test",
    size = "small",
    timeout = "moderate",
    srcs = ["model_delta_client_work_test.py"],
    deps = [
        ":client_works",
        ":model_delta_client_work",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:dataset_reduce",
        "//tensorflow_federated/python/learning/metrics:counters",
        "//tensorflow_federated/python/learning/models:functional",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:model_examples",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "proximal_client_work",
    srcs = ["proximal_client_work.py"],
    deps = [
        ":client_works",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:dataset_reduce",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/metrics:types",
        "//tensorflow_federated/python/learning/models:functional",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "proximal_client_work_test",
    size = "small",
    timeout = "moderate",
    srcs = ["proximal_client_work_test.py"],
    deps = [
        ":client_works",
        ":proximal_client_work",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:dataset_reduce",
        "//tensorflow_federated/python/learning/models:functional",
        "//tensorflow_federated/python/learning/models:model_examples",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/models:test_models",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "type_checks",
    srcs = ["type_checks.py"],
    deps = [
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "type_checks_test",
    srcs = ["type_checks_test.py"],
    deps = [
        ":type_checks",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
