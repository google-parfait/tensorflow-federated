package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:core_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/tensorflow_contet/...",

        # Impl Dependencies
        "//tensorflow_federated/python/core/impl/compiler/...",
        "//tensorflow_federated/python/core/impl/context_stack/...",

        # TODO(b/143772942): This package is temporary and can be removed once
        # all the appropriate modules have been moved out of the `impl` package.
        "//tensorflow_federated/python/core/impl",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "tensorflow_context",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)
