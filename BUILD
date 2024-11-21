load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":package_license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "package_license",
    package_name = "tensorflow_federated",
    license_kinds = ["@rules_license//licenses/spdx:Apache-2.0"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
    "README.md",
    "requirements.txt",
])

filegroup(
    name = "pyproject_toml",
    srcs = ["pyproject.toml"],
    visibility = ["//tools/python_package:python_package_tool"],
)
