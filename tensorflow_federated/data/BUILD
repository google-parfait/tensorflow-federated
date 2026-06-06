package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":data_packages",
        ":data_users",
    ],
)

package_group(
    name = "data_packages",
    packages = ["//tensorflow_federated/data/..."],
)

package_group(
    name = "data_users",
    includes = [
        "//tensorflow_federated/python:python_packages",
        "//tools:tools_packages",
    ],
)

licenses(["notice"])
