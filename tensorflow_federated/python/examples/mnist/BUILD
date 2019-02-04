package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        "//tensorflow_federated/python",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras/optimizer_v2",
    ],
)

py_test(
    name = "mnist_test",
    size = "medium",
    srcs = ["mnist_test.py"],
    deps = [
        ":mnist",
        "//tensorflow_federated/python",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
