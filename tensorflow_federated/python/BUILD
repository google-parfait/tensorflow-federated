package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# This is the public import users should use.
py_library(
    name = "python",
    srcs = ["__init__.py"],
    deps = ["//tensorflow_federated/python/core/api"],
)

py_test(
    name = "sanity_test",
    size = "small",
    srcs = ["sanity_test.py"],
    deps = [
        "//tensorflow_federated/python/core/api",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
