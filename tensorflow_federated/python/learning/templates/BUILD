load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":templates_packages",
    "//tensorflow_federated/python/learning:learning_visibility",
    "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
])

package_group(
    name = "templates_packages",
    packages = ["//tensorflow_federated/python/learning/templates/..."],
)

licenses(["notice"])

py_library(
    name = "templates",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":client_works",
        ":composers",
        ":distributors",
        ":finalizers",
        ":learning_process",
    ],
)

py_library(
    name = "client_works",
    srcs = ["client_works.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        requirement("tf-nightly"),
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
    ],
)

py_test(
    name = "client_works_test",
    size = "small",
    srcs = ["client_works_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":client_works",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
    ],
)

py_library(
    name = "composers",
    srcs = ["composers.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        ":client_works",
        ":distributors",
        ":finalizers",
        ":learning_process",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
    ],
)

py_test(
    name = "composers_test",
    size = "small",
    srcs = ["composers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":client_works",
        ":composers",
        ":distributors",
        ":finalizers",
        ":learning_process",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
    ],
)

py_library(
    name = "distributors",
    srcs = ["distributors.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":distributors",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "finalizers",
    srcs = ["finalizers.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/optimizers:keras_optimizer",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
    ],
)

py_test(
    name = "finalizers_test",
    size = "small",
    srcs = ["finalizers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":finalizers",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
    ],
)

py_library(
    name = "learning_process",
    srcs = ["learning_process.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":learning_process",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
    ],
)
