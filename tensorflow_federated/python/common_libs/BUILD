package(default_visibility = ["//tensorflow_federated/python:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# NOTE: TFF code outside this directory can depend on libraries defined in this
# directory, but not the other way round. In particular, utility classes and
# functions that depend on interfaces defined in the Core API belong in 'core',
# and those that deal with TensorFlow belong in 'tensorflow_libs'.

py_library(
    name = "py_typecheck",
    srcs = ["py_typecheck.py"],
    deps = ["@six_archive//:six"],
)

py_test(
    name = "py_typecheck_test",
    srcs = ["py_typecheck_test.py"],
    deps = [
        ":py_typecheck",
        "@six_archive//:six",
    ],
)
