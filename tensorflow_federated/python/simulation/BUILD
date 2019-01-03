# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//tensorflow_federated/python/simulation:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "simulation",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":client_data",
        ":file_per_user_client_data",
    ],
)

py_library(
    name = "client_data",
    srcs = ["client_data.py"],
    deps = ["@six_archive//:six"],
)

py_library(
    name = "file_per_user_client_data",
    srcs = ["file_per_user_client_data.py"],
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "file_per_user_client_data_test",
    srcs = ["file_per_user_client_data_test.py"],
    default_python_version = "PY3",
    deps = [
        ":file_per_user_client_data",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)
