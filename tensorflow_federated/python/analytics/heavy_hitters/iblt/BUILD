load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":iblt_packages",
        "//tensorflow_federated/python/analytics:analytics_users",

        # Python package namespace
        "//tensorflow_federated/python/analytics:__pkg__",
    ],
)

package_group(
    name = "iblt_packages",
    packages = ["//tensorflow_federated/python/analytics/heavy_hitters/iblt/..."],
)

licenses(["notice"])

py_library(
    name = "iblt",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/analytics/heavy_hitters:__pkg__"],
    deps = [
        ":chunkers",
        ":hyperedge_hashers",
        ":iblt_clipping",
        ":iblt_lib",
        ":iblt_tensor",
        ":iblt_tff",
    ],
)

py_library(
    name = "iblt_clipping",
    srcs = ["iblt_clipping.py"],
    srcs_version = "PY3",
    deps = [
        ":iblt_factory",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/analytics:data_processing",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
    ],
)

py_test(
    name = "iblt_clipping_test",
    srcs = ["iblt_clipping_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":iblt_clipping",
        ":iblt_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "iblt_lib",
    srcs = [
        "iblt_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":chunkers",
        ":hyperedge_hashers",
    ],
)

py_library(
    name = "chunkers",
    srcs = [
        "chunkers.py",
    ],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_library(
    name = "hyperedge_hashers",
    srcs = [
        "hyperedge_hashers.py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "iblt_lib_test",
    size = "medium",
    timeout = "long",
    srcs = ["iblt_lib_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [":iblt_lib"],
)

py_test(
    name = "chunkers_test",
    srcs = ["chunkers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":chunkers"],
)

py_test(
    name = "hyperedge_hashers_test",
    size = "medium",
    srcs = ["hyperedge_hashers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":hyperedge_hashers"],
)

py_library(
    name = "iblt_tff",
    srcs = ["iblt_tff.py"],
    srcs_version = "PY3",
    deps = [
        ":chunkers",
        ":iblt_lib",
        ":iblt_tensor",
        "//tensorflow_federated/python/analytics:data_processing",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "iblt_tff_test",
    timeout = "long",
    srcs = ["iblt_tff_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":iblt_tff",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
    ],
)

py_library(
    name = "iblt_tensor",
    srcs = ["iblt_tensor.py"],
    srcs_version = "PY3",
    deps = [
        ":chunkers",
        ":iblt_lib",
    ],
)

py_test(
    name = "iblt_tensor_test",
    srcs = ["iblt_tensor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":iblt_lib",
        ":iblt_tensor",
    ],
)

py_library(
    name = "iblt_factory",
    srcs = ["iblt_factory.py"],
    deps = [
        ":chunkers",
        ":iblt_tensor",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/analytics:data_processing",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "iblt_factory_test",
    srcs = ["iblt_factory_test.py"],
    deps = [
        ":chunkers",
        ":iblt_factory",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:secure",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)
