load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":analytics_packages",
    ":analytics_visibility",
])

package_group(
    name = "analytics_packages",
    packages = ["//tensorflow_federated/python/analytics/..."],
)

package_group(
    name = "analytics_visibility",
    includes = ["//tensorflow_federated/python/simulation:simulation_packages"],
)

licenses(["notice"])

py_library(
    name = "analytics",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":data_processing",
        ":histogram_processing",
        "//tensorflow_federated/python/analytics/heavy_hitters",
        "//tensorflow_federated/python/analytics/hierarchical_histogram:hierarchical_histogram_lib",
    ],
)

py_library(
    name = "data_processing",
    srcs = ["data_processing.py"],
    srcs_version = "PY3",
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)

py_test(
    name = "data_processing_test",
    srcs = ["data_processing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_processing",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "histogram_test_utils",
    testonly = True,
    srcs = ["histogram_test_utils.py"],
    srcs_version = "PY3",
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)

py_library(
    name = "histogram_processing",
    srcs = ["histogram_processing.py"],
    srcs_version = "PY3",
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)

py_test(
    name = "histogram_processing_test",
    srcs = ["histogram_processing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":histogram_processing",
        ":histogram_test_utils",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
