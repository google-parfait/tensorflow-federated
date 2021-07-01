load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/learning:learning_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/learning/framework/...",

        # TODO(b/151441025): This package is temporary and can be removed once the
        # dependencies between `tff.learning` and `tff.learning.framework` are
        # cleaned up.
        "//tensorflow_federated/python/learning/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "framework",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":encoding_utils",
        ":optimizer_utils",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_library(
    name = "client_works",
    srcs = ["client_works.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_test(
    name = "client_works_test",
    size = "small",
    srcs = ["client_works_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_works",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_library(
    name = "composers",
    srcs = ["composers.py"],
    srcs_version = "PY3",
    deps = [
        ":client_works",
        ":distributors",
        ":finalizers",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/learning:learning_process",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_test(
    name = "composers_test",
    size = "small",
    srcs = ["composers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_works",
        ":composers",
        ":distributors",
        ":finalizers",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:learning_process",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_library(
    name = "distributors",
    srcs = ["distributors.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:placements",
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
        ":distributors",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "encoding_utils",
    srcs = ["encoding_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_cpu_gpu_test(
    name = "encoding_utils_test",
    srcs = ["encoding_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":encoding_utils",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_examples",
    ],
)

py_library(
    name = "finalizers",
    srcs = ["finalizers.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_test(
    name = "finalizers_test",
    size = "small",
    srcs = ["finalizers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":finalizers",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_cpu_gpu_test(
    name = "optimizer_utils_test",
    srcs = ["optimizer_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer_utils",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:sampling",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_library(
    name = "dataset_reduce",
    srcs = ["dataset_reduce.py"],
    srcs_version = "PY3",
)

py_cpu_gpu_test(
    name = "dataset_reduce_test",
    srcs = ["dataset_reduce_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":dataset_reduce"],
)
