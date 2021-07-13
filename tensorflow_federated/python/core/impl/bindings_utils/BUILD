load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":bindings_utils_packages",
    "//tensorflow_federated/python/core/impl:impl_visibility",
    "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
    "//tensorflow_federated/python/core/impl/executors:executors_packages",
])

package_group(
    name = "bindings_utils_packages",
    packages = ["//tensorflow_federated/python/core/impl/bindings_utils/..."],
)

licenses(["notice"])

py_library(
    name = "data_conversions",
    srcs = ["data_conversions.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:placements"],
)

py_test(
    name = "data_conversions_test",
    srcs = ["data_conversions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_conversions",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "value_serialization",
    srcs = ["value_serialization.py"],
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/executors:executor_utils",
        "//tensorflow_federated/python/core/impl/executors:serialization_bindings",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "value_serialization_test",
    size = "small",
    srcs = ["value_serialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        ":value_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/executors:serialization_bindings",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)
