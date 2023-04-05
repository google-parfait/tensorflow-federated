load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "heavy_hitters_packages",
    packages = ["//tensorflow_federated/python/analytics/heavy_hitters/..."],
)

licenses(["notice"])

py_library(
    name = "heavy_hitters",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/analytics:__pkg__"],
    deps = ["//tensorflow_federated/python/analytics/heavy_hitters/iblt"],
)
