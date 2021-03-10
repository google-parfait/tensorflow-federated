load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core:core_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/utils/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":computation_utils",
        ":encoding_utils",
        ":federated_aggregations",
        ":tf_computation_utils",
    ],
)

py_library(
    name = "computation_utils",
    srcs = ["computation_utils.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "computation_utils_test",
    size = "small",
    srcs = ["computation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        ":computation_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "encoding_utils",
    srcs = ["encoding_utils.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        requirement("dm-tree"),
        requirement("tensorflow-model-optimization"),
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "encoding_utils_test",
    srcs = ["encoding_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tensorflow-model-optimization"),
        ":encoding_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "federated_aggregations",
    srcs = ["federated_aggregations.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:value_base",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "federated_aggregations_test",
    timeout = "long",
    srcs = ["federated_aggregations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        ":federated_aggregations",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/test:static_assert",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tf_computation_utils",
    srcs = ["tf_computation_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tf_computation_utils_test",
    size = "small",
    srcs = ["tf_computation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tf_computation_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:test_case",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
