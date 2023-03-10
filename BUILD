load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":package_license"],
    default_visibility = ["//visibility:private"],
)

license(
    name = "package_license",
    package_name = "tensorflow_federated",
)

licenses(["notice"])

exports_files(["LICENSE"])
