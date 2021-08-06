load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":reconstruction_packages",
    "//tensorflow_federated/python/learning:learning_visibility",
])

package_group(
    name = "reconstruction_packages",
    packages = ["//tensorflow_federated/python/learning/reconstruction/..."],
)

licenses(["notice"])

py_library(
    name = "reconstruction",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":evaluation_computation",
        ":keras_utils",
        ":model",
        ":reconstruction_utils",
        ":training_process",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
)

py_library(
    name = "reconstruction_utils",
    srcs = ["reconstruction_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_test(
    name = "reconstruction_utils_test",
    srcs = ["reconstruction_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":reconstruction_utils",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/learning:keras_utils",
    ],
)

py_test(
    name = "keras_utils_test",
    srcs = ["keras_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":model",
    ],
)

py_library(
    name = "training_process",
    srcs = ["training_process.py"],
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":model",
        ":reconstruction_utils",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/learning/optimizers:keras_optimizer",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "training_process_test",
    timeout = "long",
    srcs = ["training_process_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":model",
        ":reconstruction_utils",
        ":training_process",
        "//tensorflow_federated/python/aggregators:differential_privacy",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:robust",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "evaluation_computation",
    srcs = ["evaluation_computation.py"],
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":reconstruction_utils",
        ":training_process",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
    ],
)

py_test(
    name = "evaluation_computation_test",
    srcs = ["evaluation_computation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":evaluation_computation",
        ":keras_utils",
        ":model",
        ":reconstruction_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@absl_py//absl/testing:parameterized",
    ],
)
