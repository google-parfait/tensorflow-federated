package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "impl_packages",
    packages = [
        "//tensorflow_federated/cc/core/impl/...",
        "//tensorflow_federated/python/core/impl/...",
    ],
)

package_group(
    name = "impl_users",
    includes = ["//tensorflow_federated/cc/simulation:simulation_packages"],
)

licenses(["notice"])
