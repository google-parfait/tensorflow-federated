load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = [
    ":heavy_hitters_packages",
    "//tensorflow_federated/python/analytics:analytics_visibility",
])

package_group(
    name = "heavy_hitters_packages",
    packages = ["//tensorflow_federated/python/analytics/heavy_hitters/..."],
)

licenses(["notice"])

py_library(
    name = "heavy_hitters",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/analytics:__pkg__"],
    deps = [
        "//tensorflow_federated/python/analytics/heavy_hitters/iblt",
    ],
)
