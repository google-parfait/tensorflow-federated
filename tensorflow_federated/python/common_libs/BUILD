# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//tensorflow_federated/python:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# NOTE: TFF code outside this directory can depend on libraries defined in this
# directory, but not the other way round. In particular, utility classes and
# functions that depend on interfaces defined in the Core API belong in 'core',
# and those that deal with TensorFlow belong in 'tensorflow_libs'.

py_library(
    name = "anonymous_tuple",
    srcs = ["anonymous_tuple.py"],
    deps = [
        ":py_typecheck",
        "@six_archive//:six",
    ],
)

py_test(
    name = "anonymous_tuple_test",
    size = "small",
    srcs = ["anonymous_tuple_test.py"],
    deps = [
        ":anonymous_tuple",
        "@absl_py//absl/testing:absltest",
    ],
)

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
        "@absl_py//absl/testing:absltest",
        "@six_archive//:six",
    ],
)

py_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
