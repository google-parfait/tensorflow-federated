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
    deps = [
        ":iblt_factory",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/analytics:data_processing",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "iblt_clipping_test",
    srcs = ["iblt_clipping_test.py"],
    deps = [
        ":iblt_clipping",
        ":iblt_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "iblt_lib",
    srcs = ["iblt_lib.py"],
    deps = [
        ":chunkers",
        ":hyperedge_hashers",
    ],
)

py_library(
    name = "chunkers",
    srcs = ["chunkers.py"],
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_library(
    name = "hyperedge_hashers",
    srcs = ["hyperedge_hashers.py"],
)

py_test(
    name = "iblt_lib_test",
    size = "medium",
    timeout = "long",
    srcs = ["iblt_lib_test.py"],
    shard_count = 4,
    deps = [":iblt_lib"],
)

py_test(
    name = "chunkers_test",
    srcs = ["chunkers_test.py"],
    deps = [":chunkers"],
)

py_test(
    name = "hyperedge_hashers_test",
    size = "medium",
    srcs = ["hyperedge_hashers_test.py"],
    deps = [":hyperedge_hashers"],
)

py_library(
    name = "iblt_tff",
    srcs = ["iblt_tff.py"],
    deps = [
        ":chunkers",
        ":iblt_lib",
        ":iblt_tensor",
        "//tensorflow_federated/python/analytics:data_processing",
        "//tensorflow_federated/python/core/backends/mapreduce:intrinsics",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "iblt_tff_test",
    timeout = "long",
    srcs = ["iblt_tff_test.py"],
    shard_count = 10,
    deps = [
        ":iblt_tff",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "iblt_tensor",
    srcs = ["iblt_tensor.py"],
    deps = [
        ":chunkers",
        ":iblt_lib",
    ],
)

py_test(
    name = "iblt_tensor_test",
    srcs = ["iblt_tensor_test.py"],
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
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
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
        "@federated_language//federated_language",
    ],
)
