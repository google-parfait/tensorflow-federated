load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":iblt_packages",
    "//tensorflow_federated/python/analytics:analytics_visibility",
])

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
        ":iblt_lib",
        ":iblt_tff",
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
    deps = [
        ":iblt_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "chunkers_test",
    srcs = ["chunkers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":chunkers",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "hyperedge_hashers_test",
    size = "medium",
    srcs = ["hyperedge_hashers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hyperedge_hashers",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "iblt_tff",
    srcs = ["iblt_tff.py"],
    srcs_version = "PY3",
    deps = [
        ":chunkers",
        ":iblt_lib",
        "//tensorflow_federated/python/analytics:data_processing",
        "//tensorflow_federated/python/analytics:histogram_processing",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
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
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@absl_py//absl/testing:parameterized",
    ],
)
