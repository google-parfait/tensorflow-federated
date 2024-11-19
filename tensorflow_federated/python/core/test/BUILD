load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":test_packages",
        "//tensorflow_federated/python/core:core_users",
    ],
)

package_group(
    name = "test_packages",
    packages = ["//tensorflow_federated/python/core/test/..."],
)

licenses(["notice"])

py_library(
    name = "test",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = ["@federated_language//federated_language"],
)
